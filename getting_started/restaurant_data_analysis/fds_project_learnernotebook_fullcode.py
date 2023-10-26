import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('foodhub_order.csv')
df.head()

print("there are", df.shape[0], "rows, and", df.shape[1], "columns in our dataset")

df.info()

print('NA values\n')
df.isna().sum()

# here we are limiting the display of floating point from scientific notation to 2 dec placees
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df.describe().T

(df['rating'] == 'Not given').sum()

df.hist(figsize=(10, 20));

# create the ordered list of categories
order = df['cuisine_type'].value_counts().index
# display. the 'order' param does the lifting for us in ordering indexes
sns.countplot(x='cuisine_type', data=df, order=order)
plt.xticks(rotation=90)
plt.show()

sns.boxplot(x='cuisine_type', y='cost_of_the_order', data=df, order=order)
plt.xticks(rotation=90)
plt.show()

sns.histplot(x='cost_of_the_order', stat='count', data=df)
plt.xticks(rotation=90)
plt.show()

box_lower_bound = 10
box_higher_bound = 40

# creates a 2x2 grid and explicitly states the size of frame
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# First row: food_preparation_time
# ax is 'axis' aka index
sns.histplot(data=df,x='food_preparation_time',stat='count', ax=axs[0, 0])
sns.boxplot(data=df,x='food_preparation_time', ax=axs[0, 1])

# Second row: delivery_time
sns.histplot(data=df,x='delivery_time',stat='count', ax=axs[1, 0])
sns.boxplot(data=df,x='delivery_time', ax=axs[1, 1])

axs[0, 1].set_xlim([box_lower_bound, box_higher_bound])
axs[1, 1].set_xlim([box_lower_bound, box_higher_bound])

plt.tight_layout()
plt.show()

sns.countplot(x="food_preparation_time", hue="day_of_the_week", data=df);

top_5_values = df['restaurant_name'].value_counts().head(5)
sns.barplot(x=top_5_values.index, y=top_5_values.values)

plt.xlabel('values')
plt.ylabel('count')
plt.title('top 5 restaurants')
plt.xticks(rotation=90)
plt.show()

sns.countplot(x="cuisine_type", hue="day_of_the_week", data=df);
plt.xticks(rotation=90)
plt.show()

percentage = (df['cost_of_the_order'] > 20).mean() * 100
print(f"{percentage:.2f}% of the orders cost more than 20 dollars.")

percentage = df['delivery_time'].mean() * 100
print(f"{percentage:.2f} is the mean order delivery time.")

# Count the frequency of each customer_id in the filtered dataframe
customer_counts = df['customer_id'].value_counts()

# Get the top 3 customers who placed the most orders
customer_counts.head(3)

sns.boxplot(data=df,x='rating',y='cost_of_the_order')
plt.show()

sns.catplot(x='cost_of_the_order',
            col='cuisine_type',
            data=df,
            col_wrap=4,
            kind="violin")
plt.show()

df = df[df['rating'] != 'Not given']
df['rating'] = df['rating'].astype(float)

restaurant_group = df.groupby('restaurant_name')['rating'].agg(['sum', 'mean'])
promotional_offer_restaurants = restaurant_group[(restaurant_group['sum'] > 50) & (restaurant_group['mean'] > 4)]

promotional_offer_restaurants

# we need to get the order that have cost > 20. get sum. multiply by 0.25
# we isolated the rows that meet our condition then use 2nd param to get isolate column
loc_big_order = df.loc[df['cost_of_the_order'] > 20, 'cost_of_the_order']
loc_big_order_mod_sum = loc_big_order.sum() * 1.25

# get orders that cost < 20 but > 5. sum and multiple
loc_med_order_first = df.loc[df['cost_of_the_order'] > 5, 'cost_of_the_order']
loc_med_order_second = df.loc[df['cost_of_the_order'] < 5, 'cost_of_the_order']
loc_med_order_mod_sum = loc_med_order_second.sum() * 1.15

new_revenue = loc_big_order_mod_sum + loc_med_order_mod_sum
print(f"Total revenue is ${new_revenue:.2f}")

orders_sum = df.shape[0]
total_time_df = pd.DataFrame(
    {
        'Cooking Time': df['food_preparation_time'],
        'Delivery Time': df['delivery_time'],
        'Total Time': df['food_preparation_time'] + df['delivery_time']
    }
)
long_time = total_time_df.loc[total_time_df['Total Time'] > 60, 'Total Time'].count()
percentage = (long_time / orders_sum) * 100

print(f"{percentage}% of our order take more than an hour to get delivered")

grouped = df.groupby('day_of_the_week')['delivery_time'].agg(['median'])
print(grouped)
