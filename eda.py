# 1. Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Loading the DataSet
df = pd.read_csv("ECommerce_consumer behaviour.csv")

# 3. Number of Rows and Columns
print("Shape of DataFrame:", df.shape)

# 4. DataFrame Information
print("\nData Info:")
print(df.info())

# 5. Checking missing values
print("\nMissing Values:\n", df.isna().sum())

# 6. Checking duplicate values
print("\nDuplicate Rows:", df.duplicated().sum())

# 7. Number of unique values
print("\nUnique Values:\n", df.nunique())

# 8. Handle missing values in days_since_prior_order
df["days_since_prior_order"] = df["days_since_prior_order"].fillna(0)

# 9. Converting days_since_prior_order into int
df["days_since_prior_order"] = df["days_since_prior_order"].astype(int)

# 10. Group by order_dow (Day of the week)
dow_counts = df.groupby("order_dow")["user_id"].agg(["count"]).sort_values(by="count", ascending=False)
print("\nOrders by Day of Week:\n", dow_counts)

# 11. Number of purchases by day (Pie Plot)
dow_counts.plot(kind="pie", y="count", autopct="%1.2f%%", legend=False, title="Number of purchases by day", figsize=(8, 8))
plt.show()

# 12. Time of day when order was made
result1 = df.groupby("order_hour_of_day", as_index=False).agg({"user_id": "count"}).sort_values(by="user_id", ascending=False)
print("\nOrders by Hour of Day:\n", result1)

# 13. Categorize into time buckets
def order_time(x):
    if 6 <= x <= 12:
        return "Morning"
    elif 13 <= x <= 17:
        return "Afternoon"
    elif 18 <= x <= 22:
        return "Evening"
    else:
        return "Night"

df["order_time_list"] = df["order_hour_of_day"].apply(order_time)

# 14. Time when order made by day and time
result2 = df.groupby("order_time_list")["user_id"].agg(["count"]).sort_values(by="count", ascending=False)
print("\nOrders by Time of Day:\n", result2)
result2.plot(kind="bar", title="Time of Day Orders Made")
plt.show()

pivot = df.pivot_table(index="order_dow", columns="order_time_list", values="user_id", aggfunc="count")
pivot.plot(kind="bar", figsize=(9, 9), title="Orders by Day & Time of Day")
plt.show()

# 15. Number of orders users have made
def order_number_group(x):
    if x <= 20:
        return "1-20 orders"
    elif x <= 40:
        return "21-40 orders"
    elif x <= 60:
        return "41-60 orders"
    elif x <= 80:
        return "61-80 orders"
    else:
        return "81-100 orders"

df["order_number_group"] = df["order_number"].apply(order_number_group)
order_counts = df.groupby("order_number_group")["user_id"].agg(["count"]).sort_values(by="count", ascending=False)
print("\nOrders by User Count Grouped:\n", order_counts)
order_counts.plot(kind="pie", y="count", autopct="%1.3f%%", legend=False, title="Number of Orders", figsize=(8, 8))
plt.show()

# 16. Days since prior order
result4 = df.groupby("days_since_prior_order")["user_id"].count().sort_values(ascending=False)
print("\nDays Since Prior Order:\n", result4.head())
result4.plot(kind="bar", title="Days Since Prior Order", figsize=(9, 9))
plt.show()

# 17. Top 15 most popular products (by ID and Name)
top_products = df.groupby("product_id")["user_id"].agg(["count"]).sort_values(by="count", ascending=False).head(15)
top_products.plot(kind="pie", y="count", autopct="%1.2f%%", legend=False, title="Top 15 Products by ID", figsize=(9, 9))
plt.show()

top_products_name = df.groupby("product_name")["user_id"].agg(["count"]).sort_values(by="count", ascending=False).head(15)
top_products_name.plot(kind="pie", y="count", autopct="%1.2f%%", legend=False, title="Top 15 Products by Name", figsize=(9, 9))
plt.show()

# 18. Products users add to cart
atco_u = df.groupby("add_to_cart_order", as_index=False)["user_id"].count().sort_values(by="user_id", ascending=False)
sns.histplot(atco_u.query("user_id < 50").user_id, kde=False)
plt.title("User Count < 50 (Cart Order)")
plt.show()

# 19. Number of products added per order
on_atco = df.groupby("order_number", as_index=False)["add_to_cart_order"].count().sort_values(by="add_to_cart_order", ascending=False)
sns.histplot(on_atco["add_to_cart_order"], kde=False)
plt.title("Number of Products Added to Cart")
plt.show()

# 20. Reordered counts
print("\nReordered Value Counts:\n", df["reordered"].value_counts())
reordered_by_dept = df.groupby("department")["reordered"].count().sort_values(ascending=False)
reordered_by_dept.plot(kind="barh", title="Reorders by Department")
plt.show()

reordered_pie = df["reordered"].value_counts()
reordered_pie.plot(kind="pie", autopct="%1.2f%%", title="Reorder Distribution", figsize=(9, 9))
plt.ylabel("")
plt.show()

# 21. Top 10 reordered products
top_reorders = df.groupby("product_id")["reordered"].count().sort_values(ascending=False).head(10)
top_reorders.plot(kind="pie", y="count", autopct="%1.3f%%", legend=False, title="Top 10 Reordered Products", figsize=(9, 9))
plt.show()
