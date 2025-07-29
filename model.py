import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib
df = pd.read_csv('train.csv')
corr_matrix = df.corr()
last_col = corr_matrix.columns[-1]
cor_with_last = corr_matrix[last_col].drop(last_col)


selected = cor_with_last[abs(cor_with_last) > 0.5].index.tolist()
filtered_df = df[selected+ ['battery_power', 'px_width','px_height']]
x_train, x_test, y_train, y_test = train_test_split(filtered_df, df['price_range'], test_size=0.2, random_state=42)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(x_train, y_train)

y_hat = model.predict(x_test)
accuracy = accuracy_score(y_test, y_hat)
joblib.dump(model, 'logistic_model.pkl')