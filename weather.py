import numpy as np

# -----------------------------------
# WEATHER TEMPERATURE ANALYSIS
# -----------------------------------

# Temperature data for 30 days
temp = np.array([
    32, 34, 35, 33, 31,
    30, 29, 28, 27, 31,
    33, 35, 36, 37, 38,
    39, 40, 36, 35, 34,
    33, 32, 31, 30, 29,
    28, 27, 26, 25, 24
])

print("------ WEATHER REPORT ------")

# -----------------------------------
# TOTAL DAYS
# -----------------------------------

print("\nTotal Days:", temp.size)

# -----------------------------------
# AVERAGE TEMPERATURE
# -----------------------------------

avg_temp = np.mean(temp)

print("\nAverage Temperature:", round(avg_temp, 2), "°C")

# -----------------------------------
# HOTTEST DAY
# -----------------------------------

max_temp = np.max(temp)

hot_day = np.argmax(temp) + 1

print("\nHighest Temperature:", max_temp, "°C")
print("Hottest Day:", hot_day)

# -----------------------------------
# COLDEST DAY
# -----------------------------------

min_temp = np.min(temp)

cold_day = np.argmin(temp) + 1

print("\nLowest Temperature:", min_temp, "°C")
print("Coldest Day:", cold_day)

# -----------------------------------
# DAYS ABOVE 35°C
# -----------------------------------

hot_days = temp[temp > 35]

print("\nTemperatures Above 35°C:")
print(hot_days)

# -----------------------------------
# DAYS BELOW 30°C
# -----------------------------------

cold_days = temp[temp < 30]

print("\nTemperatures Below 30°C:")
print(cold_days)

# -----------------------------------
# WEEKLY ANALYSIS
# -----------------------------------

weeks = temp.reshape(6, 5)

print("\nWeekly Temperature Data:")
print(weeks)

# -----------------------------------
# WEEKLY AVERAGE
# -----------------------------------

weekly_avg = np.mean(weeks, axis=1)

print("\nWeekly Average Temperature:")
print(weekly_avg)

# -----------------------------------
# TEMPERATURE DIFFERENCE
# -----------------------------------

difference = np.max(temp) - np.min(temp)

print("\nTemperature Difference:", difference, "°C")

# -----------------------------------
# SORTED TEMPERATURE
# -----------------------------------

sorted_temp = np.sort(temp)

print("\nSorted Temperatures:")
print(sorted_temp)

# -----------------------------------
# FINAL MESSAGE
# -----------------------------------

print("\nWeather Analysis Completed")