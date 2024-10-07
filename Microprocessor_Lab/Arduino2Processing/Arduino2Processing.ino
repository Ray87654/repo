const int xAxisPin = A0;  // X軸連接到類比輸入0
const int yAxisPin = A1;  // Y軸連接到類比輸入1
const int numRows = 1;  // 按鈕陣列的行數
const int numCols = 2;  // 按鈕陣列的列數
int rowPins[numRows] = {2};  // 修改為實際連接的引腳
int colPins[numCols] = {3,4};  // 修改為實際連接的引腳


void setup() {
  Serial.begin(9600);
  
  // 設置第一行引腳為輸出並初始化為高電位
  pinMode(rowPins[0], OUTPUT);
  digitalWrite(rowPins[0], HIGH);

  // 設置列引腳為輸入並啟用內部上拉電阻
  for (int i = 0; i < numCols; i++) {
    pinMode(colPins[i], INPUT_PULLUP);
  }
}

void loop() {
  int A = 0;
  digitalWrite(rowPins[0], LOW);

  for (int j = 0; j < 2; j++) {
    int buttonState = digitalRead(colPins[j]);
    // 如果按鈕 1 被按下，輸出 0
    if (j == 0 && buttonState == LOW) 
    {
      A = 1;
    }

    if (j == 1 && buttonState == LOW) 
    {
      A = 2;
    }
  }
  digitalWrite(rowPins[0], HIGH);
  Serial.println(A);
  delay(5);  // 等待一段時間，以免打印太快
}
