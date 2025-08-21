// main.ino — Nano 33 BLE Sense Rev2: BLE/Serial time sync + TFLM inference + notify 5 floats
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include <Arduino_HS300x.h>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <stdint.h>

#include "model.h"                 // defines: extern const uint8_t g_model[]; extern const int g_model_len;
#include "normalization_data.h"    // generated header (namespace: norm, with NormalizeX/DenormalizeY)

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// Optional: set your local timezone offset in SECONDS (e.g. CET=+3600, CEST=+7200)
static const int32_t kTzOffsetSeconds = 0;  // change if you want local time

// ---------- TFLM globals ----------
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ---- BLE UUIDs (Nordic UART-like) ----
const char* kServiceUUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
const char* kTXCharUUID  = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"; // notify (peripheral -> laptop)
const char* kRXCharUUID  = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"; // write  (laptop -> peripheral)

BLEService        svc(kServiceUUID);
// TX sends 5 floats: [day, year, temp, hum, y] => 20 bytes
BLECharacteristic txChar(kTXCharUUID, BLERead | BLENotify, sizeof(float) * 5);
// RX receives time sync (8-byte epoch_ms LE) or ASCII "TIME <ms>\n"
BLECharacteristic rxChar(kRXCharUUID, BLEWrite | BLEWriteWithoutResponse, 20);
}  // namespace

// ---------- Time sync state & helpers ----------
static uint64_t g_epoch_ms_base = 0;  // host epoch (ms) at last sync
static uint32_t g_millis_base   = 0;  // millis() at last sync
static bool     g_has_time      = false;

static void UpdateTimeSync(uint64_t host_epoch_ms) {
  g_epoch_ms_base = host_epoch_ms;
  g_millis_base   = millis();
  g_has_time      = true;
}

static uint64_t NowEpochMs() {
  if (!g_has_time) return 0;
  return g_epoch_ms_base + (uint32_t)(millis() - g_millis_base);
}

// Prefer real-time when available; fallback to uptime-based since-boot fractions.
static float day_fraction() {
  if (g_has_time) {
    const double sec = NowEpochMs() / 1000.0;
    const double day = 86400.0;
    double f = fmod(sec, day) / day;
    if (f < 0) f += 1.0;
    return (float)f;
  } else {
    const float sec = millis() / 1000.0f;
    const float day = 86400.0f;
    float f = fmodf(sec, day) / day;
    if (f < 0) f += 1.0f;
    return f;
  }
}

static float year_fraction() {
  if (g_has_time) {
    const double sec = NowEpochMs() / 1000.0;
    const double yr  = 365.25 * 86400.0;
    double f = fmod(sec, yr) / yr;
    if (f < 0) f += 1.0;
    return (float)f;
  } else {
    const float sec = millis() / 1000.0f;
    const float yr  = 365.25f * 86400.0f;
    float f = fmodf(sec, yr) / yr;
    if (f < 0) f += 1.0f;
    return f;
  }
}

static void get_hour_minute(uint8_t& hh, uint8_t& mm) {
  uint64_t sec;
  if (g_has_time) {
    sec = NowEpochMs() / 1000ULL;       // real UTC seconds since epoch
  } else {
    sec = millis() / 1000UL;            // fallback: uptime seconds
  }
  sec += (uint64_t)kTzOffsetSeconds;    // apply timezone if desired

  const uint32_t sod = (uint32_t)(sec % 86400UL); // seconds-of-day
  hh = (uint8_t)(sod / 3600U);
  mm = (uint8_t)((sod % 3600U) / 60U);
}

// ---------- Setup ----------
void setup() {
  tflite::InitializeTarget();

  Serial.begin(115200);
  while (!Serial) {}

  // BLE init
  if (!BLE.begin()) {
    Serial.println("BLE.begin() failed");
  } else {
    BLE.setLocalName("NanoBLE-Model");
    BLE.setAdvertisedService(svc);
    svc.addCharacteristic(txChar);
    svc.addCharacteristic(rxChar);
    BLE.addService(svc);
    txChar.writeValue((const uint8_t*)nullptr, 0);
    BLE.advertise();
    Serial.println("BLE advertising (NanoBLE-Model)");
  }

  // Sensor init (HS300x on Nano 33 BLE Sense Rev2)
  if (!HS300x.begin()) {
    Serial.println("HS300x init failed (continuing)");
  }

  // TFLM init
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Schema mismatch: "); Serial.print(model->version());
    Serial.print(" vs "); Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  // Expect 4 input features
  const int rank = input->dims->size;
  const int features = input->dims->data[rank - 1];
  Serial.print("Input type="); Serial.print((int)input->type);
  Serial.print(" features=");  Serial.println(features);
  if (features != 4) {
    Serial.println("WARNING: model input features != 4");
  }
}

// ---------- Loop ----------
void loop() {
  BLE.poll(); // keep BLE stack alive

  // ---- BLE time sync: either 8-byte LE epoch_ms, or ASCII "TIME <ms>" ----
  if (rxChar.written()) {
    uint8_t buf[20];
    int n = rxChar.valueLength();
    if (n > (int)sizeof(buf)) n = sizeof(buf);
    if (rxChar.readValue(buf, n)) {
      if (n >= 8) { // binary
        uint64_t ms = 0;
        for (int i = 0; i < 8; ++i) ms |= (uint64_t)buf[i] << (8*i);
        UpdateTimeSync(ms);
        Serial.print("Time sync (BLE, ms)="); Serial.println((unsigned long)(ms/1000ULL));
      } else if (n >= 5 && !memcmp(buf, "TIME ", 5)) { // ASCII fallback
        unsigned long long ms = 0;
        for (int i = 5; i < n; ++i) {
          if (buf[i] < '0' || buf[i] > '9') break;
          ms = ms*10 + (buf[i]-'0');
        }
        UpdateTimeSync((uint64_t)ms);
        Serial.println("Time sync (BLE ASCII)");
      }
    }
  }

  // ---- Serial time sync: "TIME <epoch_ms>\n" or "TIME_S <epoch_s>\n" ----
  if (Serial.available()) {
    static char line[64];
    size_t idx = 0;
    while (Serial.available() && idx+1 < sizeof(line)) {
      char c = (char)Serial.read();
      if (c == '\n' || c == '\r') break;
      line[idx++] = c;
    }
    line[idx] = '\0';

    if (idx >= 5 && strncmp(line, "TIME ", 5) == 0) {
      unsigned long long ms = strtoull(line+5, nullptr, 10);
      UpdateTimeSync((uint64_t)ms);
      Serial.println("Time sync (SER ms)");
    } else if (idx >= 7 && strncmp(line, "TIME_S ", 7) == 0) {
      unsigned long long s = strtoull(line+7, nullptr, 10);
      UpdateTimeSync((uint64_t)s * 1000ULL);
      Serial.println("Time sync (SER s)");
    }
  }

  // ---- Read sensors ----
  float temp_c = HS300x.readTemperature();
  float hum    = HS300x.readHumidity();
  if (isnan(temp_c)) temp_c = 0.0f;
  if (isnan(hum))    hum    = 0.0f;

  // ---- Assemble features in your order: time_day, time_year, temp, hum ----
  float feats[4] = { day_fraction(), year_fraction(), temp_c, hum };

  // ---- Normalize using generated header ----
  float nx[4];
  norm::NormalizeX(feats, nx);

  // ---- Fill input ----
  if (input->type == kTfLiteInt8) {
    const float s = input->params.scale;
    const int   z = input->params.zero_point;
    for (int i = 0; i < 4; ++i) {
      int32_t q = (int32_t)lrintf(nx[i] / s) + z;
      if (q < -128) q = -128; else if (q > 127) q = 127;
      input->data.int8[i] = (int8_t)q;
    }
  } else if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < 4; ++i) input->data.f[i] = nx[i];
  } else {
    delay(200);
    return;
  }

  // ---- Inference ----
  if (interpreter->Invoke() != kTfLiteOk) {
    delay(200);
    return;
  }

  // ---- Output: denormalize to original units ----
  float y_norm = (output->type == kTfLiteInt8)
                   ? ((float)output->data.int8[0] - output->params.zero_point) * output->params.scale
                   : output->data.f[0];
  float y = norm::DenormalizeY(y_norm, 0);

  // ---- Notify 5 floats over BLE (day, year, temp, hum, y) ----
  float pkt[5] = { feats[0], feats[1], temp_c, hum, y };
  txChar.writeValue((uint8_t*)pkt, sizeof(pkt));

  // ---- Serial print (optional) ----
  Serial.print("day="); Serial.print(feats[0], 6);
  Serial.print(" year="); Serial.print(feats[1], 6);
  Serial.print(" T="); Serial.print(temp_c, 2);
  Serial.print("C H="); Serial.print(hum, 2);
  Serial.print("% -> Y="); Serial.println(y, 3);

  uint8_t hh, mm;
  get_hour_minute(hh, mm);

  // ... your existing prints ...
  Serial.print(" time=");
  if (hh < 10) Serial.print('0');
  Serial.print(hh);
  Serial.print(':');
  if (mm < 10) Serial.print('0');
  Serial.print(mm);

  // Example of full line with your existing fields:
  Serial.print(" day=");  Serial.print(feats[0], 6);
  Serial.print(" year="); Serial.print(feats[1], 6);
  Serial.print(" T=");    Serial.print(temp_c, 2);
  Serial.print("C H=");   Serial.print(hum, 2);
  Serial.print("% -> Y=");Serial.print(y, 3);
  Serial.print(" @ ");    // optional separator
  if (hh < 10) Serial.print('0'); Serial.print(hh);
  Serial.print(':');
  if (mm < 10) Serial.print('0'); Serial.println(mm);

  delay(500);
}