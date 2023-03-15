# machine-learning
topic : Company Bankruptcy Prediction
Dataset Introduction :
Bankruptcy data from the Taiwan Economic Journal for the years 1999–2009	

Technology expected to be used :
(1)	SVM : 由於此資料集為二元分類資料集，以0、1代表破產與否，因此選用在二元分類上效果優秀的分類器。
(2)	Naïve Bayes : 透過條件機率進行分類，雖然可以進行numeric 資料的訓練，但不確定其適用度。
(3)	PCA : 由於在觀察資料集過後發現資料集擁有非常多的feature 因此希望透過PCA 降低feature 的維度，以降低訓練難度，並期望透過其增加訓練的準確度。
(4)	K – Fold :透過交叉驗證，避免overfit和選擇偏差等問題，並期望能改善訓練準確度。
(5)	K – NN:以資料的距離以及鄰居個數決定新資料的屬性，但缺點是對局部性資料較為敏感。
(6)	Random Forest:使用Decision Tree集成來判斷決策的邊界，隨機森林結合了許多Decision Tree，減少overfit的風險。
(7)	Decision Tree: Decision Tree 受歡迎的原因在於，解釋容易、可處理範疇特徵、並且可延伸到多重類型的分類環境。不需要特徵縮放，也能捕捉非線性特性和特徵交互作用，和Random Forest做比較。

實驗目的 :
透過比較 SVM 、 Naïve Bayes 、 K – NN 、 Random Forest 、 Decision Tree來觀察這些模型在numeric資料集上的適用度。

實驗步驟概述 :
(1)	前處理 : 補缺漏值、進行資料集的平衡、進行PCA 降低資料集維度等。
(2)	建立模型 :建立 SVM 、 Naïve Bayes 、 K – NN 、 Random Forest 、 Decision Tree模型，並透過 K – fold 來改善準確度。
