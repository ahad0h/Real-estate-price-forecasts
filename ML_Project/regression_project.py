# مشروع تعلم الآلة - توقع أسعار العقارات

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# الخطوة 1: تحميل البيانات
data = pd.read_csv("houses.csv")
print("Dataset shape:", data.shape)
print(data.head())

# الخطوة 2: تنظيف البيانات (القيم الناقصة والتكرار)
print("\nChecking missing values:")
print(data.isnull().sum())

if data.isnull().sum().sum() > 0:
    data = data.fillna(data.median())
    print("Missing values filled with median")
else:
    print("No missing values found")

duplicates = data.duplicated().sum()
print("Duplicate rows:", duplicates)
if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicates removed")

# الخطوة 3: تحديد المتغيرات (المساحة + الغرف + الحمامات) والهدف (السعر)
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# الخطوة 4: تقسيم البيانات (80% تدريب - 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# الخطوة 5: تدريب نموذجين (الانحدار الخطي + شجرة القرار)
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# الخطوة 6: تقييم النموذجين
models = {"Linear Regression": linear_model, "Decision Tree": tree_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print("\n", name)
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    print("R²:", round(r2_score(y_test, y_pred), 4))

# الخطوة 7: تجربة توقع سعر بيت جديد
new_house = pd.DataFrame([[200, 4, 2]], columns=['area', 'bedrooms', 'bathrooms'])
print("\nPrediction for new house (200m, 3 rooms, 2 bathrooms):")
for name, model in models.items():
    pred = model.predict(new_house)
    print(f"{name}: {round(pred[0]):,} SAR")
