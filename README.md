# ECG-Classification

# Index
- 목적
- 데이터 설명
- 모델
  - CNN
    - Concept
    - Model
    - Stratified K-Fold
    - Result
  - CNN-LSTM
    - Concept
    - Time Distributed
    - Model
    - Result
  - Seq2Seq
    - Concept
    - Model
    - Result  
  
## 목적
classify the 5 different ~~~

## 데이터 설명

## 모델
**CNN**
- Concept

- Model
```
class Convnet(Model):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1_1 = Conv1D(16, kernel_size = 5, activation = 'relu', padding = 'valid')
        self.conv1_2 = Conv1D(16, kernel_size = 5, activation = 'relu', padding = 'valid')
        self.max1 = MaxPool1D(pool_size = 2)
        self.drop1 = Dropout(rate = 0.1)
        self.conv2_1 = Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid')
        self.conv2_2 = Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid')
        self.max2 = MaxPool1D(pool_size = 2)
        self.drop2 = Dropout(rate = 0.1)
        self.conv3_1 = Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid')
        self.conv3_2 = Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid')
        self.max3 = MaxPool1D(pool_size = 2)
        self.drop3 = Dropout(rate =  0.1)
        self.conv4_1 = Conv1D(256, kernel_size =3, activation = 'relu', padding = 'valid')
        self.conv4_2 = Conv1D(256, kernel_size =3, activation = 'relu', padding = 'valid')
        self.max4 = GlobalMaxPool1D()
        self.drop4 = Dropout(rate = 0.2)

        self.dense1 = Dense(64, activation = 'relu')
        self.dense2 = Dense(64, activation = 'relu')
        self.dense3 = Dense(5, activation = 'softmax')
    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max1(x)
        x = self.drop1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max2(x)
        x = self.drop2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.max3(x)
        x = self.drop3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.max4(x)
        x = self.drop4(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x 
```
- Stratified K-Fold
**K-Fold** 

데이터를 k개 분할로 나누고, k-1개의 분할에서 훈련이 이루어지며, 나머지 분할에서 평가하는 방법입니다. score의 경우, 각 Fold 별 score들의 평균값을 사용합니다.

![](https://blog.kakaocdn.net/dn/Ac7Cd/btqXXkuYgrM/k2kXdtXSpyoHuBmHVqFGw0/img.png)
(code)
```
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=2021)


<parameters>
-n_splits (int) : Fold의 개수 k 값
-shuffle (bool) : 데이터를 쪼갤 때 섞을지 유무
-random_state (int) : 내부적으로 사용되는 난수값
```

---
**Stratified K-Fold**

기존의 K-Fold의 경우, 분류할 클래스의 비율에 상관없이 데이터를 분할합니다. 하지만 분류할 클래스의 비율이 다를 경우, K-Fold를 적용한 각 Fold가 학습 dataset을 대표한다고 할 수 없습니다. 이러한 문제를 해결하기 위해 Stratified K-Fold가 제안되었습니다.

Stratified K-Fold의 경우, k개의 Fold로 분할한 이후에도 전체 훈련 데이터의 클래스 비율과 각 Fold가 가지고 있는 클래스의 비율을 맞추어 주기 때문에 dataset의 대표성이 보장됩니다. 

![](https://www.researchgate.net/profile/Mohsen-Azimi-2/publication/336889074/figure/fig18/AS:822836264460288@1573190861533/Visualization-of-stratified-k-fold-cross-validation-with-k5.png)
(code)
```
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

<parameters>
-n_splits (int) : Fold의 개수 k 값
-shuffle (bool) : 데이터를 쪼갤 때 섞을지 유무
-random_state (int) : 내부적으로 사용되는 난수값
```


```
#stratified k-fold
from sklearn.model_selection import StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
```

```
pred_list = []
for train_index, val_index in skf.split(data, label):
  x_train, y_train = data[train_index], label[train_index]
  x_valid, y_valid = data[val_index], label[val_index]
  model = Convnet()
  model.compile(optimizer=optimizer,loss= 'sparse_categorical_crossentropy', metrics=['acc'])
  model.fit(x_train, y_train, epochs = 300, batch_size = 32,  validation_data = (x_valid, y_valid), callbacks = callbacks)
  _, acc = model.evaluate(x_test, y_test)
  y_pred = model.predict(x_test) / 5
  pred_list.append(y_pred)
```

```
pred = np.array(pred_list[0])
for i in range (4):
  pred += np.array(pred_list[i+1])
  
y_pred = np.argmax(pred,axis=1)
```
- Result
```
from sklearn.metrics import accuracy_score, f1_score

f1 = f1_score(y_test, y_pred, average = 'macro')
print("Test f1 score : %s "% f1)

accuracy = accuracy_score(y_test, y_pred)
print("Test acc score : %s "% accuracy)
```

**CNN-LSTM**
- Concept


- Time Distributed

- Model
```
class Cnnlstm(Model):
    def __init__(self):
        super(Cnnlstm, self).__init__()
        self.conv1_1 = tf.keras.layers.TimeDistributed(Conv1D(16, kernel_size = 5, activation = 'relu', padding = 'valid'))
        self.conv1_2 = tf.keras.layers.TimeDistributed(Conv1D(16, kernel_size = 5, activation = 'relu', padding = 'valid'))
        self.max1 = tf.keras.layers.TimeDistributed(MaxPool1D(pool_size = 2))
        self.drop1 = tf.keras.layers.TimeDistributed(Dropout(rate = 0.1))
        self.conv2_1 = tf.keras.layers.TimeDistributed(Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.conv2_2 = tf.keras.layers.TimeDistributed(Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.max2 = tf.keras.layers.TimeDistributed(MaxPool1D(pool_size = 2))
        self.drop2 = tf.keras.layers.TimeDistributed(Dropout(rate = 0.1))
        self.conv3_1 = tf.keras.layers.TimeDistributed(Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.conv3_2 = tf.keras.layers.TimeDistributed(Conv1D(32, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.max3 = tf.keras.layers.TimeDistributed(MaxPool1D(pool_size = 2))
        self.drop3 = tf.keras.layers.TimeDistributed(Dropout(rate =  0.1))
        self.conv4_1 = tf.keras.layers.TimeDistributed(Conv1D(256, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.conv4_2 = tf.keras.layers.TimeDistributed(Conv1D(256, kernel_size =3, activation = 'relu', padding = 'valid'))
        self.fc = tf.keras.layers.TimeDistributed(Flatten())
        self.lstm1 = LSTM(78, activation = 'relu', return_sequences = True)
        self.lstm2 = LSTM(11, activation = 'relu', return_sequences = True)
        self.dense1 = Dense(75, activation = 'relu')
        self.dense2= Dense(5, activation = 'softmax')


       
    def call(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max1(x)
        x = self.drop1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max2(x)
        x = self.drop2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.max3(x)
        x = self.drop3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.fc(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x 
```

- Result

