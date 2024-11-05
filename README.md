## 機台軸承異常分析：探索式數據分析與機器學習練習作品 EDA and ML for Machine Bearing Defects
1. 探討異常與各部件振動間之相互關係，以作為異常改善的方向參考，同時建立分類器模型以檢出異常。
2. 使用Matplotlib與Seaborn製作圖表，以視覺化特徵之間相互關係。
3. 藉由熱力圖進行初步特徵選擇，再使用主成分分析作降維處理。
4. 使用GridSearchCV輔助模型調參。
5. 使用K-fold交叉驗證法穩定模型表現。
###
1. Explore the relationship between machine failure and component fluctuations as a reference for failure reduction, and establish a classifier model to detect failure.
2. Use Matplotlib and Seaborn visualize relationships between features.
3. Use a heatmap for feature selection, and PCA for dimensionality reduction.
4. Use GridSearchCV for hyperparameter tuning.
5. Apply K-fold cross-validation to stabilize model performance.
## 分析成果 Conclusions
1. 透過EDA，觀察到左右軸承的Y軸振動值呈現正向關聯，且能夠顯著分群正常與異常樣本。 Through EDA, a positive relationship has been found between Y-fluctuations of left and right bearings, which separates positive and negative samples significantly.
   ![image](https://github.com/user-attachments/assets/963455ab-10d5-4dcf-a5f9-79f9fdfb1361)
2. 透過熱力圖，觀察到溫度與異常間呈現低相關，因此在後續的學習模型上移除溫度特徵。 Through the heatmap, a weak relationship has been found between temperature and failure, so we removed temperature features in the following ML model.
   ![image](https://github.com/user-attachments/assets/151a3e14-cb41-4506-bf60-d29070df9acb)
3. 使用PCA將維度縮減為3，並保留90%以上的問題解釋能力。 Further reduced dimention to 3 via PCA, while retaining over 90% of the explained variance.

   ![image](https://github.com/user-attachments/assets/30d644f0-1430-4ecc-8892-3fa1014dc9be)
4. 經由GridSearchCV輔助調參後，模型表現如下。 Performance after hyperparameter tuning via GridSearchCV is as below.

   ![image](https://github.com/user-attachments/assets/b968e083-adc9-479b-b707-d18de467074c)

## 環境 Environment
* Python 3.12.4
* `pip install -r requirements.txt`
## 檔案 Files
* `Data_of_rotary_machine_defects` 存放資料集。本研究使用Kaggle開源資料集：https://www.kaggle.com/datasets/kazakovyurii/data-of-rotary-machine-defects/data
* `main.py` 主程式。
###
* `Data_of_rotary_machine_defects` stores dataset. This side project refers to Kaggle open-source dataset: https://www.kaggle.com/datasets/kazakovyurii/data-of-rotary-machine-defects/data
* `main.py` main program.
## 用法 Directions
可視需求建立虛擬環境，參考上面指令，利用 `requirements.txt` 文件安裝所需模組後，即可開啟`main.py` 檢視程式碼與執行。
###
You may set up a virtual environment if needed. Refer to the command above and install necessary modules via `requirements.txt`, and then open `main.py`.
## 授權 Autorization
MIT License
