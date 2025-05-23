/*
 * Smart Sign Language Translator Glove
 * Authors: Bhavjot Gill & Arvind Dhaliwal
 * Date: 01/12/2025 (Updated: 2025-05-21)
 *
 * - Added /get-labels endpoint to retrieve saved gesture labels.
 * - Added /delete-gesture?label=LABEL endpoint to delete a specific gesture.
 * - Added /delete-all-gestures endpoint to delete all training data.
 * - Integrated VoiceRSS Text-to-Speech for ESP32-based audio output.
 * - Plays a test tone on startup via I2S.
 * - I2S Pins: DIN (DOUT) = 25, BCLK = 26, LRC (WS) = 27
 * - Added NULL check for WiFiClient stream in speakWithVoiceRSS.
 * - Incorporated user-provided API key and WiFi credentials.
 * - Optimized speakWithVoiceRSS to reduce stack usage by making buffers static.
 * - REMOVED physical button support entirely. Web UI controls only.
 * - Added MPU6050 gyroscope calibration at startup to reduce drift.
 * - Web UI reset button calls /reset on ESP32 to clear accumulated gyro values.
 * - Made I2S initialization stricter with error checking.
 * - Ensured all function definitions are complete.
 */

#include <Arduino.h>
#include <cmath>
#include <algorithm>
#include <array>
#include <vector>
#include <LittleFS.h>
#include <WiFi.h>
#include <FS.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>
#include <Arduino_JSON.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <map>
#include <set> // For storing unique labels

// For I2S
#include "driver/i2s.h"
// For VoiceRSS HTTP Client
#include <WiFiClientSecure.h> // For HTTPS
#include <HTTPClient.h>

// --- VoiceRSS Configuration ---
#define VOICERSS_API_KEY "4191561eb8d34538a2dc0d746c5e14d5" 
const char* voicerss_host = "api.voicerss.org";
const char* voicerss_lang = "vi-vn";
const char* voicerss_voice = "Chi"; 
const char* voicerss_format = "16khz_16bit_mono"; 
const char* voicerss_codec = "WAV";

// I2S Pin Definitions
#define I2S_DOUT_PIN   25 
#define I2S_BCLK_PIN   26 
#define I2S_LRC_PIN    27 
#define I2S_SAMPLE_RATE   (16000) 
#define I2S_BITS_PER_SAMPLE (16)   
#define I2S_PLAY_BUFFER_SIZE (1024) // Size of the buffer to read from HTTP stream (mono bytes)
#define I2S_DMA_BUFFER_COUNT (8)
#define I2S_DMA_BUFFER_LEN   (256) // Frames per DMA buffer
#define I2S_NUM (I2S_NUM_0)      
#define TONE_FREQUENCY    (440)    
#define TONE_DURATION_MS  (300)

// Timer variables & counters
unsigned long lastTime = 0;
unsigned long gyroDelay = 50;
int counter = 0;
unsigned long lastFlexDebugPrintTime = 0; 

// Feature Scaling & MPU
const float FLEX_MIN_TARGET = 0.0;
const float FLEX_MAX_TARGET = 1.0; 
const float GYRO_RAW_MIN = -4.36; // Approx -250 deg/s in rad/s
const float GYRO_RAW_MAX = 4.36;  // Approx +250 deg/s in rad/s
const float ACCEL_RAW_MIN = -156.9064; // Approx -16G in m/s^2
const float ACCEL_RAW_MAX = 156.9064;  // Approx +16G in m/s^2

Adafruit_MPU6050 mpu;
sensors_event_t a, g, temp_event;
float gyroX_accumulated = 0.0, gyroY_accumulated = 0.0, gyroZ_accumulated = 0.0;
float accX_current, accY_current, accZ_current;

float gyro_bias_x = 0.0;
float gyro_bias_y = 0.0;
float gyro_bias_z = 0.0;
float gyroXerror = 0.003; 
float gyroYerror = 0.0005;
float gyroZerror = 0.00017;

// WiFi Credentials
const char *ssid = "Kien";     
const char *password = "0968404490"; 

AsyncWebServer server(80);
AsyncEventSource events("/events");

// Flex Sensor Pins & Values
const int flexPinThumb = 32;  const int flexPinIndex = 35; const int flexPinMiddle = 34; 
const int flexPinRing = 39;   const int flexPinPinky = 36;  
float flexValThumb, flexValIndex, flexValMiddle, flexValRing, flexValPinky;

// KNN Data Structures
struct DataCSV_ML { float Avals[132]; std::string label; };
DataCSV_ML Y_Traind[50];            
const int Y_Traind_Max_Size = 50;   
const char *csvFilename = "/SensorData.csv";
const char *tempCsvFilename = "/SensorData_temp.csv"; 

int currentMode = 0; 
String currentTrainingLabel = "";
int k_knn = 3; 

// State flag for TTS
volatile bool isSpeaking = false; 

// Sensor Data Queues
#define QUEUE_SIZE 12
class SensorQueue {
private:
    float queue_data[QUEUE_SIZE]; int front_idx; int rear_idx; int current_count;
public:
    SensorQueue() : front_idx(0), rear_idx(-1), current_count(0) {}
    bool enqueue(float value) {
        if (current_count >= QUEUE_SIZE) dequeue();
        rear_idx = (rear_idx + 1) % QUEUE_SIZE;
        queue_data[rear_idx] = value; current_count++; return true;
    }
    float dequeue() {
        if (current_count <= 0) return 0.0f;
        float value = queue_data[front_idx];
        front_idx = (front_idx + 1) % QUEUE_SIZE; current_count--; return value;
    }
    float getAt(int i) const {
        if (i < 0 || i >= current_count) return 0.0f; 
        return queue_data[(front_idx + i) % QUEUE_SIZE];
    }
    int getCount() const { return current_count; }
    void clearAndFillZeros() {
        front_idx = 0; rear_idx = -1; current_count = 0;
        for (int i = 0; i < QUEUE_SIZE; ++i) enqueue(0.0f);
    }
};
SensorQueue qFlexThumb, qFlexIndex, qFlexMiddle, qFlexRing, qFlexPinky;
SensorQueue qGyroX, qGyroY, qGyroZ;
SensorQueue qAccelX, qAccelY, qAccelZ;

// --- Function Prototypes ---
void initI2S();
void playStartupTone();
void speakWithVoiceRSS(String textToSpeak);
void initMPU();
void calibrateMPU();
float normalizeValue(float value, float val_min_raw, float val_max_raw, float target_min, float target_max);
void loadDataset();
void resetAndFillQueues();
void GetKNN_Predict(int k_val, const DataCSV_ML& newDataInstance);

// --- Function Definitions ---
void initI2S() {
    Serial.println("Cấu hình I2S...");
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, 
        .dma_buf_count = I2S_DMA_BUFFER_COUNT,
        .dma_buf_len = I2S_DMA_BUFFER_LEN,
        .use_apll = false, 
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK_PIN, .ws_io_num = I2S_LRC_PIN,
        .data_out_num = I2S_DOUT_PIN, .data_in_num = I2S_PIN_NO_CHANGE
    };

    esp_err_t err = i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("LỖI NGHIÊM TRỌNG: Cài đặt trình điều khiển I2S thất bại! Mã lỗi: %d. Hệ thống dừng.\n", err);
        while(1) { delay(1000); } 
    }
    err = i2s_set_pin(I2S_NUM, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("LỖI NGHIÊM TRỌNG: Đặt chân I2S thất bại! Mã lỗi: %d. Gỡ cài đặt trình điều khiển.\n", err);
        i2s_driver_uninstall(I2S_NUM); 
        while(1) { delay(1000); } 
    }
    err = i2s_zero_dma_buffer(I2S_NUM);
    if (err != ESP_OK) {
        Serial.printf("LỖI: Không thể xóa bộ đệm DMA I2S! Mã lỗi: %d.\n", err);
    }
    Serial.println("I2S đã được cấu hình thành công.");
}

void playStartupTone() {
    Serial.println("Đang phát âm thanh khởi động...");
    static int16_t buffer[I2S_DMA_BUFFER_LEN * 2]; 
    size_t bytes_written; 
    int total_frames_for_tone = (I2S_SAMPLE_RATE * TONE_DURATION_MS) / 1000;

    isSpeaking = true; 
    for (int frame_offset = 0; frame_offset < total_frames_for_tone; frame_offset += I2S_DMA_BUFFER_LEN) {
        int frames_in_this_chunk = std::min((int)I2S_DMA_BUFFER_LEN, total_frames_for_tone - frame_offset);
        for (int i = 0; i < frames_in_this_chunk; ++i) {
            double time_sec = (double)(frame_offset + i) / I2S_SAMPLE_RATE;
            int16_t sample = (int16_t)(INT16_MAX * 0.08 * sin(2 * PI * TONE_FREQUENCY * time_sec)); 
            buffer[i * 2] = sample; buffer[i * 2 + 1] = sample; 
        }
        esp_err_t write_err = i2s_write(I2S_NUM, buffer, frames_in_this_chunk * 2 * (I2S_BITS_PER_SAMPLE / 8), &bytes_written, pdMS_TO_TICKS(1000));
         if (write_err != ESP_OK) {
            Serial.printf("Lỗi ghi I2S khi phát tone: %d\n", write_err);
        }
    }
    i2s_zero_dma_buffer(I2S_NUM); 
    isSpeaking = false;
    Serial.println("Âm thanh khởi động đã phát xong.");
}

void speakWithVoiceRSS(String textToSpeak) {
    if (isSpeaking) {
        Serial.println("VoiceRSS: Đang bận phát âm thanh khác, yêu cầu TTS bị bỏ qua.");
        return;
    }
    isSpeaking = true;

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("VoiceRSS: Không có kết nối WiFi để thực hiện TTS.");
        isSpeaking = false;
        return;
    }
    if (strcmp(VOICERSS_API_KEY, "YOUR_VOICERSS_API_KEY_HERE") == 0 || strlen(VOICERSS_API_KEY) < 10) {
        Serial.println("VoiceRSS: API Key chưa được cấu hình đúng. Vui lòng cập nhật VOICERSS_API_KEY.");
        isSpeaking = false;
        return;
    }

    Serial.printf("VoiceRSS: Yêu cầu TTS cho: \"%s\"\n", textToSpeak.c_str());

    HTTPClient http;
    WiFiClientSecure client_secure; 
    client_secure.setInsecure(); 

    String url = "https://" + String(voicerss_host) + "/?key=" + String(VOICERSS_API_KEY) +
                 "&hl=" + String(voicerss_lang) +
                 "&v=" + String(voicerss_voice) +
                 "&src=" + textToSpeak +
                 "&f=" + String(voicerss_format) + 
                 "&c=" + String(voicerss_codec);
    url.replace(" ", "%20"); 

    Serial.println("VoiceRSS URL: " + url);

    if (http.begin(client_secure, url)) {
        int httpCode = http.GET();
        if (httpCode > 0) {
            Serial.printf("VoiceRSS: Mã phản hồi HTTP: %d\n", httpCode);
            if (httpCode == HTTP_CODE_OK) {
                WiFiClient* stream = http.getStreamPtr();
                if (stream == nullptr) {
                    Serial.println("VoiceRSS: Lỗi - Không nhận được stream audio từ HTTPClient.");
                } else {
                    static char header_buffer[44]; 
                    int header_bytes_to_skip = 44; 
                    int skipped_count = 0;
                    unsigned long header_timeout_start = millis();
                    while(skipped_count < header_bytes_to_skip && (millis() - header_timeout_start < 5000)) {
                        if (stream->connected() && stream->available()) {
                            int bytes_to_read_now = std::min(header_bytes_to_skip - skipped_count, (int)stream->available());
                            bytes_to_read_now = std::min(bytes_to_read_now, (int)sizeof(header_buffer));
                            stream->readBytes(header_buffer, bytes_to_read_now);
                            skipped_count += bytes_to_read_now;
                        } else if (!stream->connected()) {
                            Serial.println("VoiceRSS: Stream ngắt kết nối khi đang đọc header WAV.");
                            break;
                        }
                        delay(1);
                    }

                    if(skipped_count < header_bytes_to_skip) {
                        Serial.println("VoiceRSS: Lỗi đọc hết header WAV hoặc hết thời gian chờ.");
                    } else {
                        Serial.println("VoiceRSS: Đã bỏ qua header WAV. Bắt đầu phát audio...");
                        static uint8_t mono_http_buffer[I2S_PLAY_BUFFER_SIZE]; 
                        static int16_t stereo_i2s_buffer[I2S_PLAY_BUFFER_SIZE]; 
                        size_t bytes_written_to_i2s;
                        int total_audio_bytes_played = 0;
                        unsigned long audio_read_timeout_start = millis();

                        while (http.connected() && (millis() - audio_read_timeout_start < 20000)) { 
                            if (stream->available()) {
                                audio_read_timeout_start = millis(); 
                                size_t mono_bytes_read = stream->readBytes(mono_http_buffer, sizeof(mono_http_buffer));
                                if (mono_bytes_read > 0) {
                                    if (mono_bytes_read % 2 != 0) {
                                        Serial.println("VoiceRSS: Đọc được số byte lẻ cho audio mono, bỏ qua frame cuối.");
                                        mono_bytes_read--; 
                                        if (mono_bytes_read == 0) continue;
                                    }
                                    int num_mono_samples = mono_bytes_read / 2; 
                                    if (num_mono_samples > I2S_PLAY_BUFFER_SIZE / 2) { 
                                        Serial.println("VoiceRSS: Lỗi logic - num_mono_samples quá lớn cho stereo_i2s_buffer.");
                                        break;
                                    }
                                    for(int i=0; i < num_mono_samples; ++i) {
                                        int16_t mono_sample = (int16_t)(mono_http_buffer[i*2] | (mono_http_buffer[i*2+1] << 8));
                                        stereo_i2s_buffer[i*2]     = mono_sample; 
                                        stereo_i2s_buffer[i*2 + 1] = mono_sample; 
                                    }
                                    esp_err_t i2s_write_err = i2s_write(I2S_NUM, stereo_i2s_buffer, num_mono_samples * 4, &bytes_written_to_i2s, pdMS_TO_TICKS(1000));
                                    if(i2s_write_err != ESP_OK){
                                        Serial.printf("VoiceRSS: Lỗi ghi I2S: %d\n", i2s_write_err);
                                        break; 
                                    }
                                    total_audio_bytes_played += mono_bytes_read; 
                                } else if (mono_bytes_read < 0) {
                                    Serial.println("VoiceRSS: Lỗi đọc stream audio.");
                                    break;
                                }
                            } else if (!http.connected()) { 
                                Serial.println("VoiceRSS: HTTP server ngắt kết nối khi đang stream audio.");
                                break;
                            }
                            delay(1); 
                        }
                        i2s_zero_dma_buffer(I2S_NUM); 
                        Serial.printf("VoiceRSS: Phát audio hoàn tất. Tổng số byte audio (mono) đã nhận: %d\n", total_audio_bytes_played);
                    } 
                } 
            } else if (httpCode == HTTP_CODE_SERVICE_UNAVAILABLE || httpCode == HTTP_CODE_FORBIDDEN || httpCode == HTTP_CODE_UNAUTHORIZED) {
                 String payload = http.getString();
                 Serial.printf("VoiceRSS: Lỗi dịch vụ, API Key không hợp lệ hoặc không được phép. Phản hồi: %s\n", payload.c_str());
            } else {
                Serial.printf("VoiceRSS: Yêu cầu GET thất bại, mã lỗi HTTP: %d\n", httpCode);
                String payload = http.getString();
                Serial.println("Phản hồi lỗi từ server: " + payload);
            }
        } else {
            Serial.printf("VoiceRSS: Yêu cầu GET thất bại, lỗi HTTPClient: %s\n", http.errorToString(httpCode).c_str());
        }
        http.end(); 
    } else {
        Serial.printf("VoiceRSS: Không thể bắt đầu kết nối tới %s\n", voicerss_host);
    }
    isSpeaking = false; 
}

void initMPU() {
    if (!mpu.begin()) {
        Serial.println("Lỗi: Không tìm thấy chip MPU6050. Hệ thống dừng.");
        while (1) delay(10);
    }
    Serial.println("Đã tìm thấy MPU6050!");
    mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    Serial.println("MPU6050 đã cấu hình: Gia tốc +/-16G, Con quay +/-250deg/s.");
}

void calibrateMPU() {
    Serial.println("Đang hiệu chỉnh con quay hồi chuyển MPU6050. Vui lòng giữ thiết bị đứng yên...");
    delay(1000); 

    long num_samples = 1000; 
    double gx_sum = 0, gy_sum = 0, gz_sum = 0;

    Serial.print("Đang lấy mẫu hiệu chỉnh: ");
    for (int i = 0; i < num_samples; i++) {
        sensors_event_t accel_event, gyro_event, temp_event_calib;
        mpu.getEvent(&accel_event, &gyro_event, &temp_event_calib);
        gx_sum += gyro_event.gyro.x;
        gy_sum += gyro_event.gyro.y;
        gz_sum += gyro_event.gyro.z;
        if (i % (num_samples/10) == 0) { 
             Serial.print(".");
        }
        delay(3); 
    }
    Serial.println(" Hoàn tất!");

    gyro_bias_x = gx_sum / num_samples;
    gyro_bias_y = gy_sum / num_samples;
    gyro_bias_z = gz_sum / num_samples;

    Serial.println("Hiệu chỉnh con quay hồi chuyển hoàn tất.");
    Serial.printf("  Thiên vị Gyro X: %.5f rad/s\n", gyro_bias_x);
    Serial.printf("  Thiên vị Gyro Y: %.5f rad/s\n", gyro_bias_y);
    Serial.printf("  Thiên vị Gyro Z: %.5f rad/s\n", gyro_bias_z);
}

float normalizeValue(float value, float val_min_raw, float val_max_raw, float target_min, float target_max) {
    if (val_max_raw == val_min_raw) return target_min;
    float normalized = (value - val_min_raw) / (val_max_raw - val_min_raw);
    normalized = constrain(normalized, 0.0, 1.0); 
    return normalized * (target_max - target_min) + target_min;
}

void loadDataset() {
    File file = LittleFS.open(csvFilename, "r");
    for(int i=0; i<Y_Traind_Max_Size; ++i) Y_Traind[i].label = ""; // Clear Y_Traind before loading

    if (!file || file.isDirectory()) { 
        if (file) file.close(); 
        Serial.printf("Lỗi: Không thể mở tệp %s để đọc hoặc nó là một thư mục. Không có dữ liệu huấn luyện nào được tải.\n", csvFilename);
        return;
    }
    Serial.printf("Đang tải bộ dữ liệu từ %s...\n", csvFilename);
    int row = 0;
    while (file.available() && row < Y_Traind_Max_Size) {
        String line = file.readStringUntil('\n'); line.trim();
        if (line.length() == 0) continue;
        int col = 0; char lineCopy[line.length() + 1]; strcpy(lineCopy, line.c_str());
        char *token = strtok(lineCopy, ",");
        while (token != nullptr && col < 133) {
            if (col < 132) Y_Traind[row].Avals[col] = atof(token);
            else Y_Traind[row].label = std::string(token);
            token = strtok(nullptr, ","); col++;
        }
        if (col == 133 && !Y_Traind[row].label.empty()) row++;
        else {
            Serial.printf("Cảnh báo: Dòng %d trong bộ dữ liệu bị lỗi hoặc không hoàn chỉnh: %s\n", row + 1, line.c_str());
            Y_Traind[row].label = ""; 
        }
    }
    Serial.printf("Đã tải %d mục hợp lệ vào Y_Traind.\n", row);
    file.close();
}

void resetAndFillQueues() {
    qFlexThumb.clearAndFillZeros(); qFlexIndex.clearAndFillZeros(); qFlexMiddle.clearAndFillZeros();
    qFlexRing.clearAndFillZeros(); qFlexPinky.clearAndFillZeros();
    qGyroX.clearAndFillZeros(); qGyroY.clearAndFillZeros(); qGyroZ.clearAndFillZeros();
    qAccelX.clearAndFillZeros(); qAccelY.clearAndFillZeros(); qAccelZ.clearAndFillZeros();
}

void GetKNN_Predict(int k_val, const DataCSV_ML& newDataInstance) {
    if (isSpeaking) {
        Serial.println("KNN: Đang bận nói, bỏ qua dự đoán lần này.");
        return;
    }
    Serial.println("--- KNN Prediction ---");
    Serial.print("Input Data Snippet (first value of each queue: P,R,M,I,T, Gx,Gy,Gz, Ax,Ay,Az): ");
    SensorQueue* debug_queues[] = {&qFlexPinky, &qFlexRing, &qFlexMiddle, &qFlexIndex, &qFlexThumb,
                                 &qGyroX, &qGyroY, &qGyroZ, &qAccelX, &qAccelY, &qAccelZ};
    for(int i=0; i<11; ++i) { 
        if (debug_queues[i]->getCount() > 0) { 
             Serial.printf("%.3f,", debug_queues[i]->getAt(debug_queues[i]->getCount() - 1) ); 
        } else {
             Serial.print("N/A,");
        }
    } 
    Serial.println();

    int validTrainingSamples = 0;
    for(int i=0; i < Y_Traind_Max_Size; ++i) {
        if(!Y_Traind[i].label.empty()) {
            validTrainingSamples++;
        }
    }

    if (validTrainingSamples == 0) {
        Serial.println("Lỗi KNN: Dữ liệu huấn luyện chưa được tải hoặc trống.");
        JSONVar errorMsg; errorMsg["error"] = "Dữ liệu huấn luyện chưa được tải hoặc trống.";
        events.send(JSON.stringify(errorMsg).c_str(), "prediction_error", millis());
        currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0; 
        return;
    }

    std::vector<std::pair<float, std::string>> distances; 

    for (int i = 0; i < Y_Traind_Max_Size; ++i) {
        if (Y_Traind[i].label.empty()) continue; 

        double current_distance_sq = 0; 
        for (int j = 0; j < 132; ++j) {
            float diff = newDataInstance.Avals[j] - Y_Traind[i].Avals[j];
            current_distance_sq += diff * diff;
        }
        distances.push_back({(float)current_distance_sq, Y_Traind[i].label}); 
    }

    if (distances.empty()) {
        Serial.println("Lỗi KNN: Không có khoảng cách nào được tính toán.");
        currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0; 
        return;
    }
    
    int effective_k = std::min(k_val, (int)distances.size());
    if (effective_k == 0) {
        Serial.println("Lỗi KNN: Effective k là 0.");
        currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0; 
        return;
    }

    std::partial_sort(distances.begin(), distances.begin() + effective_k, distances.end());

    std::map<std::string, int> labelCounts;
    for (int i = 0; i < effective_k; ++i) {
        labelCounts[distances[i].second]++;
    }

    std::string mostCommonLabel = ""; 
    int maxCount = 0;
    if (!labelCounts.empty()) {
        for (const auto &pair : labelCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                mostCommonLabel = pair.first;
            }
        }
    } else {
        Serial.println("Cảnh báo KNN: Không tìm thấy nhãn nào trong số K láng giềng.");
        currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0; return;
    }

    float confidence = static_cast<float>(maxCount) / effective_k;
    Serial.printf("Nhãn dự đoán: %s (Độ tin cậy: %.2f%%)\n", mostCommonLabel.c_str(), confidence * 100);
    Serial.println("  Top K labels:");
    for(int i=0; i<effective_k; ++i) {
        Serial.printf("    %d. %s (Dist^2: %.4f)\n", i+1, distances[i].second.c_str(), distances[i].first);
    }

    JSONVar predictionJson; 
    predictionJson["pred"] = mostCommonLabel.c_str();
    predictionJson["confidence"] = confidence;

    String lowerLabel = mostCommonLabel.c_str(); 
    lowerLabel.toLowerCase();

    float speakConfidenceThreshold = 0.50; 

    if (confidence >= speakConfidenceThreshold) { 
        if (lowerLabel != "at rest" && lowerLabel != " nghỉ") { 
            events.send(JSON.stringify(predictionJson).c_str(), "prediction", millis()); 
            Serial.printf("Đã gửi dự đoán tới UI (tin cậy >= %.0f%%): %s\n", speakConfidenceThreshold*100, mostCommonLabel.c_str());
            speakWithVoiceRSS(mostCommonLabel.c_str()); 
        } else {
            events.send(JSON.stringify(predictionJson).c_str(), "prediction", millis()); 
            Serial.println("Dự đoán 'At rest' hoặc 'Nghỉ', không phát âm thanh TTS.");
        }
    } else { 
        Serial.println("Độ tin cậy dưới ngưỡng phát âm. Gửi dưới dạng sự kiện 'low_confidence' tới UI.");
        events.send(JSON.stringify(predictionJson).c_str(), "low_confidence", millis());
    }
    currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0;
}

void setup() {
    delay(1000); 
    Serial.begin(115200);
    Serial.println("\nĐang khởi tạo Găng Tay Dịch Ngôn Ngữ Ký Hiệu Thông Minh...");

    WiFi.begin(ssid, password);
    Serial.print("Đang kết nối WiFi");
    for(int retries = 0; WiFi.status() != WL_CONNECTED && retries < 20; ++retries) {
        delay(500); Serial.print(".");
    }
    if(WiFi.status() == WL_CONNECTED) {
        Serial.println("\nĐã kết nối WiFi");
        Serial.print("Địa chỉ IP: "); Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nLỗi kết nối WiFi. Giao diện web và TTS có thể không hoạt động.");
    }

    initMPU();
    calibrateMPU(); 
    initI2S(); 

    if (!LittleFS.begin(true)) { 
        Serial.println("Lỗi nghiêm trọng: Khởi tạo LittleFS thất bại! Không thể lưu/tải dữ liệu."); 
        return; 
    }
    Serial.println("LittleFS đã khởi tạo.");
    
    if (!LittleFS.exists(csvFilename)) {
        Serial.printf("Tệp %s không tồn tại. Đang tạo tệp mới...\n", csvFilename);
        File dataFile = LittleFS.open(csvFilename, "w");
        if (dataFile) {
            dataFile.close();
            Serial.printf("Đã tạo tệp %s.\n", csvFilename);
        } else {
            Serial.printf("Lỗi: Không thể tạo tệp %s.\n", csvFilename);
        }
    }
    
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){ request->send(LittleFS, "/index.html", "text/html"); });
    server.on("/style.css", HTTP_GET, [](AsyncWebServerRequest *request){ request->send(LittleFS, "/style.css", "text/css"); });

    server.on("/reset", HTTP_GET, [](AsyncWebServerRequest *request){ 
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh.");
            return;
        }
        gyroX_accumulated = 0; gyroY_accumulated = 0; gyroZ_accumulated = 0;
        Serial.println("Đã đặt lại hướng Gyroscope (qua web).");
        request->send(200, "text/plain", "OK: Gyroscope reset");
    });
    server.on("/predict", HTTP_GET, [](AsyncWebServerRequest *request){ 
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh, vui lòng thử lại sau.");
            return;
        }
        Serial.println("Yêu cầu web: Chuyển sang Chế độ Dự đoán (mode 1)");
        currentMode = 1; counter = 0; gyroDelay = 250; 
        loadDataset(); resetAndFillQueues();
        request->send(200, "text/plain", "OK: Chế độ dự đoán đã kích hoạt. Thực hiện cử chỉ.");
        JSONVar statusMsg; statusMsg["status"] = "capturing_prediction";
        events.send(JSON.stringify(statusMsg).c_str(), "system_status", millis());
    });
    server.on("/train", HTTP_GET, [](AsyncWebServerRequest *request){ 
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh, vui lòng thử lại sau.");
            return;
        }
        if (request->hasParam("label")) {
            currentTrainingLabel = request->getParam("label")->value();
            if (currentTrainingLabel.length() > 0 && currentTrainingLabel.indexOf(',') == -1) {
                Serial.printf("Yêu cầu web: Chuyển sang Chế độ Huấn luyện (mode 2) cho nhãn: %s\n", currentTrainingLabel.c_str());
                currentMode = 2; counter = 0; gyroDelay = 250;
                resetAndFillQueues();
                request->send(200, "text/plain", "OK: Bắt đầu ghi huấn luyện cho nhãn: " + currentTrainingLabel + ". Thực hiện cử chỉ.");
            } else {
                request->send(400, "text/plain", "Lỗi: Nhãn không hợp lệ (ví dụ: trống hoặc chứa dấu phẩy).");
            }
        } else {
            request->send(400, "text/plain", "Lỗi: Thiếu tham số 'label'.");
        }
    });
    
    server.on("/get-labels", HTTP_GET, [](AsyncWebServerRequest *request){
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh.");
            return;
        }
        Serial.println("Yêu cầu web: Lấy danh sách nhãn đã lưu.");
        
        if (!LittleFS.exists(csvFilename)) {
             Serial.printf("Tệp %s không tồn tại khi lấy nhãn.\n", csvFilename);
             JSONVar emptyArray; 
             request->send(200, "application/json", JSON.stringify(emptyArray));
             return;
        }

        File file = LittleFS.open(csvFilename, "r");
        if (!file || file.isDirectory()) {
            if(file) file.close();
            Serial.printf("Lỗi: Không thể mở tệp %s để đọc danh sách nhãn.\n", csvFilename);
            request->send(500, "application/json", "{\"error\":\"Không thể mở tệp dữ liệu\"}");
            return;
        }

        std::set<std::string> uniqueLabels; 
        int lineCount = 0;
        while (file.available()) {
            String line = file.readStringUntil('\n');
            lineCount++;
            line.trim();
            if (line.length() > 0) {
                int lastComma = line.lastIndexOf(',');
                if (lastComma != -1 && lastComma < (int)line.length() - 1) {
                    String labelStr = line.substring(lastComma + 1);
                    labelStr.trim(); 
                    if (labelStr.length() > 0) {
                         uniqueLabels.insert(labelStr.c_str());
                    } else {
                        Serial.printf("Cảnh báo: Nhãn rỗng ở dòng %d trong %s\n", lineCount, csvFilename);
                    }
                } else {
                     Serial.printf("Cảnh báo: Định dạng dòng %d không đúng trong %s (thiếu dấu phẩy cuối cho nhãn?): %s\n", lineCount, csvFilename, line.c_str());
                }
            }
        }
        file.close();

        JSONVar labelsArray;
        int index = 0;
        for (const std::string& label : uniqueLabels) {
            labelsArray[index++] = label.c_str();
        }
        
        String jsonResponse = JSON.stringify(labelsArray);
        request->send(200, "application/json", jsonResponse);
        Serial.printf("Đã gửi %d nhãn duy nhất: %s\n", uniqueLabels.size(), jsonResponse.c_str());
    });

    server.on("/delete-gesture", HTTP_GET, [](AsyncWebServerRequest *request){
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh.");
            return;
        }
        if (request->hasParam("label")) {
            String labelToDelete = request->getParam("label")->value();
            labelToDelete.trim();
            Serial.printf("Yêu cầu web: Xóa cử chỉ có nhãn: '%s'\n", labelToDelete.c_str());

            File originalFile = LittleFS.open(csvFilename, "r");
            if (!originalFile) {
                Serial.printf("Lỗi: Không thể mở tệp %s để xóa nhãn.\n", csvFilename);
                request->send(500, "application/json", "{\"error\":\"Không thể mở tệp dữ liệu\"}");
                return;
            }

            File tempFile = LittleFS.open(tempCsvFilename, "w");
            if (!tempFile) {
                Serial.printf("Lỗi: Không thể tạo tệp tạm thời %s.\n", tempCsvFilename);
                originalFile.close();
                request->send(500, "application/json", "{\"error\":\"Không thể tạo tệp tạm thời\"}");
                return;
            }

            bool foundAndDeleted = false;
            while (originalFile.available()) {
                String line = originalFile.readStringUntil('\n');
                String currentLabel = "";
                int lastComma = line.lastIndexOf(',');
                if (lastComma != -1 && lastComma < (int)line.length() - 1) {
                    currentLabel = line.substring(lastComma + 1);
                    currentLabel.trim();
                }

                if (currentLabel.equalsIgnoreCase(labelToDelete)) {
                    foundAndDeleted = true;
                    Serial.printf("  Đã tìm thấy và bỏ qua dòng cho nhãn: %s\n", labelToDelete.c_str());
                } else {
                    tempFile.println(line);
                }
            }
            originalFile.close();
            tempFile.close();

            if (foundAndDeleted) {
                if (!LittleFS.remove(csvFilename)) {
                    Serial.printf("Lỗi: Không thể xóa tệp gốc %s.\n", csvFilename);
                    request->send(500, "application/json", "{\"error\":\"Lỗi xóa tệp gốc\"}");
                    return;
                }
                if (!LittleFS.rename(tempCsvFilename, csvFilename)) {
                    Serial.printf("Lỗi: Không thể đổi tên tệp tạm thời thành %s.\n", csvFilename);
                    request->send(500, "application/json", "{\"error\":\"Lỗi đổi tên tệp tạm thời\"}");
                    return;
                }
                Serial.printf("Đã xóa thành công tất cả các mục cho nhãn '%s'.\n", labelToDelete.c_str());
                loadDataset(); 
                request->send(200, "application/json", "{\"success\":true, \"message\":\"Đã xóa cử chỉ.\"}");
            } else {
                LittleFS.remove(tempCsvFilename); 
                Serial.printf("Không tìm thấy cử chỉ với nhãn '%s' để xóa.\n", labelToDelete.c_str());
                request->send(404, "application/json", "{\"error\":\"Không tìm thấy cử chỉ\"}");
            }
        } else {
            request->send(400, "application/json", "{\"error\":\"Thiếu tham số 'label'\"}");
        }
    });

    server.on("/delete-all-gestures", HTTP_GET, [](AsyncWebServerRequest *request){
        if (isSpeaking) {
            request->send(503, "text/plain", "Lỗi: Thiết bị đang bận phát âm thanh.");
            return;
        }
        Serial.println("Yêu cầu web: Xóa tất cả cử chỉ đã lưu.");
        if (LittleFS.remove(csvFilename)) {
            Serial.printf("Đã xóa thành công tệp %s.\n", csvFilename);
            for(int i=0; i<Y_Traind_Max_Size; ++i) Y_Traind[i].label = "";
            File dataFile = LittleFS.open(csvFilename, "w"); // Recreate empty file
            if (dataFile) dataFile.close();

            request->send(200, "application/json", "{\"success\":true, \"message\":\"Đã xóa tất cả cử chỉ.\"}");
        } else {
            Serial.printf("Lỗi: Không thể xóa tệp %s. Có thể tệp không tồn tại.\n", csvFilename);
             for(int i=0; i<Y_Traind_Max_Size; ++i) Y_Traind[i].label = ""; 
            request->send(200, "application/json", "{\"success\":true, \"message\":\"Không có cử chỉ nào để xóa hoặc đã xóa.\"}");
        }
    });
    
    events.onConnect([](AsyncEventSourceClient *client){
        if (client->lastId()) Serial.printf("SSE Client đã kết nối lại! ID tin nhắn cuối cùng: %u\n", client->lastId());
        client->send("hello!", NULL, millis(), 10000);
    });
    server.addHandler(&events);
    server.begin();
    Serial.println("Máy chủ HTTP đã bắt đầu.");

    currentMode = 0; gyroDelay = 50; resetAndFillQueues(); 
    Serial.println("Thiết lập hoàn tất. Thiết bị đang ở Chế độ Rảnh/Quan sát (mode 0).");

    playStartupTone(); 
    // speakWithVoiceRSS("Hệ thống đã khởi động"); 
}

void loop() {
    // --- FLEX SENSOR CALIBRATION ---
    // !!! QUAN TRỌNG: THAY THẾ CÁC GIÁ TRỊ NÀY BẰNG GIÁ TRỊ THỰC NGHIỆM CỦA BẠN !!!
    // Đây chỉ là ví dụ, bạn PHẢI tự đo cho từng cảm biến với điện trở 4.7k Ohm.
    // Ghi lại giá trị analogRead() thấp nhất (thẳng) và cao nhất (cong) cho mỗi ngón.
    int adc_min_straight_thumb = 400;  int adc_max_bent_thumb = 2800;  // THAY THẾ!
    int adc_min_straight_index = 450;  int adc_max_bent_index = 2900;  // THAY THẾ!
    int adc_min_straight_middle= 420;  int adc_max_bent_middle= 2750; // THAY THẾ!
    int adc_min_straight_ring  = 480;  int adc_max_bent_ring  = 3000;  // THAY THẾ!
    int adc_min_straight_pinky = 500;  int adc_max_bent_pinky = 3100;  // THAY THẾ!
    
    int rawFlexThumb = analogRead(flexPinThumb);
    int rawFlexIndex = analogRead(flexPinIndex);
    int rawFlexMiddle = analogRead(flexPinMiddle);
    int rawFlexRing = analogRead(flexPinRing);
    int rawFlexPinky = analogRead(flexPinPinky);

    flexValThumb  = normalizeValue(rawFlexThumb,  adc_min_straight_thumb, adc_max_bent_thumb, 0.0, 1.0); 
    flexValIndex  = normalizeValue(rawFlexIndex,  adc_min_straight_index, adc_max_bent_index, 0.0, 1.0); 
    flexValMiddle = normalizeValue(rawFlexMiddle, adc_min_straight_middle,adc_max_bent_middle,0.0, 1.0); 
    flexValRing   = normalizeValue(rawFlexRing,   adc_min_straight_ring,  adc_max_bent_ring,  0.0, 1.0); 
    flexValPinky  = normalizeValue(rawFlexPinky,  adc_min_straight_pinky, adc_max_bent_pinky, 0.0, 1.0); 

    if (millis() - lastFlexDebugPrintTime > 2000) { 
       Serial.printf("Raw Flex (T,I,M,R,P): %d, %d, %d, %d, %d\n", rawFlexThumb, rawFlexIndex, rawFlexMiddle, rawFlexRing, rawFlexPinky);
       Serial.printf("Norm Flex (T,I,M,R,P): %.2f, %.2f, %.2f, %.2f, %.2f\n", flexValThumb, flexValIndex, flexValMiddle, flexValRing, flexValPinky);
       lastFlexDebugPrintTime = millis();
    }

    mpu.getEvent(&a, &g, &temp_event);
    accX_current = a.acceleration.x; accY_current = a.acceleration.y; accZ_current = a.acceleration.z;
    
    float gyroX_rate_calibrated = g.gyro.x - gyro_bias_x;
    float gyroY_rate_calibrated = g.gyro.y - gyro_bias_y;
    float gyroZ_rate_calibrated = g.gyro.z - gyro_bias_z;

    float dt = (float)gyroDelay / 1000.0; 
    if (abs(gyroX_rate_calibrated) > gyroXerror) gyroX_accumulated += gyroX_rate_calibrated * dt * (180.0 / PI);
    if (abs(gyroY_rate_calibrated) > gyroYerror) gyroY_accumulated += gyroY_rate_calibrated * dt * (180.0 / PI);
    if (abs(gyroZ_rate_calibrated) > gyroZerror) gyroZ_accumulated += gyroZ_rate_calibrated * dt * (180.0 / PI);

    JSONVar currentReadingsJson;
    currentReadingsJson["thumb"] = String(flexValThumb, 3); currentReadingsJson["Index"] = String(flexValIndex, 3);
    currentReadingsJson["Middle"] = String(flexValMiddle, 3); currentReadingsJson["Ring"] = String(flexValRing, 3);
    currentReadingsJson["Pinky"] = String(flexValPinky, 3);
    currentReadingsJson["accX"] = String(accX_current, 3); currentReadingsJson["accY"] = String(accY_current, 3);
    currentReadingsJson["accZ"] = String(accZ_current, 3);
    currentReadingsJson["gyroX"] = String(gyroX_accumulated, 2); currentReadingsJson["gyroY"] = String(gyroY_accumulated, 2);
    currentReadingsJson["gyroZ"] = String(gyroZ_accumulated, 2);
    String jsonStringToClient = JSON.stringify(currentReadingsJson);

    if ((millis() - lastTime) > gyroDelay) {
        lastTime = millis();
        if (WiFi.status() == WL_CONNECTED) {
            if (currentMode == 0 ) events.send(jsonStringToClient.c_str(), "flex_sensor", millis());
            else if (counter % 5 == 0) events.send(jsonStringToClient.c_str(), "flex_sensor", millis());
        }

        qFlexThumb.enqueue(flexValThumb); qFlexIndex.enqueue(flexValIndex); qFlexMiddle.enqueue(flexValMiddle);
        qFlexRing.enqueue(flexValRing); qFlexPinky.enqueue(flexValPinky);
        
        qGyroX.enqueue(normalizeValue(gyroX_rate_calibrated, GYRO_RAW_MIN, GYRO_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        qGyroY.enqueue(normalizeValue(gyroY_rate_calibrated, GYRO_RAW_MIN, GYRO_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        qGyroZ.enqueue(normalizeValue(gyroZ_rate_calibrated, GYRO_RAW_MIN, GYRO_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        
        qAccelX.enqueue(normalizeValue(accX_current, ACCEL_RAW_MIN, ACCEL_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        qAccelY.enqueue(normalizeValue(accY_current, ACCEL_RAW_MIN, ACCEL_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        qAccelZ.enqueue(normalizeValue(accZ_current, ACCEL_RAW_MIN, ACCEL_RAW_MAX, FLEX_MIN_TARGET, FLEX_MAX_TARGET));
        
        static DataCSV_ML currentGestureInstance; 
        
        SensorQueue* queues[] = {&qFlexPinky, &qFlexRing, &qFlexMiddle, &qFlexIndex, &qFlexThumb,
                                 &qGyroX, &qGyroY, &qGyroZ, &qAccelX, &qAccelY, &qAccelZ};
        for (int stream_idx = 0; stream_idx < 11; ++stream_idx) {
            for (int time_step = 0; time_step < QUEUE_SIZE; ++time_step) {
                currentGestureInstance.Avals[time_step + (stream_idx*QUEUE_SIZE)] = queues[stream_idx]->getAt(time_step);
            }
        }

        if (currentMode == 1 && !isSpeaking) { 
            if (counter == 0) {
                Serial.println("Chế độ Dự đoán: Bắt đầu ghi cử chỉ... Thực hiện ký hiệu.");
            }
            counter++;
            if (counter >= QUEUE_SIZE) {
                Serial.println("Ghi cử chỉ hoàn tất để dự đoán. Đang chạy KNN...");
                GetKNN_Predict(k_knn, currentGestureInstance); 
            }
        }
        else if (currentMode == 2 && !isSpeaking) { 
             if (counter == 0) {
                Serial.printf("Chế độ Huấn luyện: Bắt đầu ghi cho nhãn '%s'. Thực hiện ký hiệu.\n", currentTrainingLabel.c_str());
            }
            counter++;
            if (counter >= QUEUE_SIZE) {
                currentGestureInstance.label = currentTrainingLabel.c_str();
                File dataFile = LittleFS.open(csvFilename, "a");
                if (!dataFile) {
                    Serial.printf("Lỗi: Không thể mở %s để ghi thêm!\n", csvFilename);
                    JSONVar tUpd; tUpd["status"] = "error_saving"; tUpd["label"] = currentTrainingLabel;
                    tUpd["message"] = "Không thể mở tệp dữ liệu."; events.send(JSON.stringify(tUpd).c_str(), "training_update", millis());
                } else {
                    String dataRow = "";
                    for (int i = 0; i < 132; ++i) { dataRow += String(currentGestureInstance.Avals[i], 6); dataRow += ","; }
                    dataRow += currentGestureInstance.label.c_str();
                    dataFile.println(dataRow); dataFile.close();
                    Serial.printf("Dữ liệu cho nhãn '%s' đã được lưu vào %s\n", currentGestureInstance.label.c_str(), csvFilename);
                    JSONVar tUpd; tUpd["status"] = "saved"; tUpd["label"] = currentTrainingLabel; 
                    events.send(JSON.stringify(tUpd).c_str(), "training_update", millis());
                }
                currentMode = 0; gyroDelay = 50; resetAndFillQueues(); counter = 0; currentTrainingLabel = ""; 
                Serial.println("Ghi huấn luyện hoàn tất. Đã trở lại Chế độ Rảnh/Quan sát.");
                JSONVar sMsg; sMsg["status"] = "idle_after_training"; events.send(JSON.stringify(sMsg).c_str(), "system_status", millis());
            }
        }
    }
}
