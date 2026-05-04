increase_sales_percent = 12.93720081
increase_revenue_percent = 18.33206078

formatted_sales_percent = "{:.2f}".format(increase_sales_percent)
formatted_revenue_percent = "{:.2f}".format(increase_revenue_percent)

str = f"In the last year, sales were up by {formatted_sales_percent}% and revenue went up by {formatted_revenue_percent}%."
print(str)
