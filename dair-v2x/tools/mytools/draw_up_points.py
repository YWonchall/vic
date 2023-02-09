import matplotlib.pyplot as plt


plt.figure(figsize=(20, 10), dpi=100)
plt.xlabel("cost/Byte")
plt.ylabel("mAP")
plt.title("Trend of MAP with the number of point cloud fusion")

# pp
# cost = [0,4570,13567,32422,53742,72456,114633,170560,225051,433990,650351,955363]
# y_05 = [59.04,60.51,64.91,66.84,68.03,68.28,68.58,68.65,68.59,69.09,69.46,69.56]
# y_03 = [63.50,65.27,69.11,70.27,70.86,70.92,71.49,71.56,71.56,71.81,72.13,72.17]
# y_07 = [51.69,52.22,55.65,58.08,58.48,59.16,59.87,60.15,59.83,60.22,60.53,60.24]

# sp
# y_03 = [63.50,64.72,68.59,69.20,70.11,70.10,70.36,70.54,70.54,70.32,70.50,70.92,71.27,72.17]
# y_05 = [59.04,59.81,64.49,65.41,66.90,66.90,66.93,67.23,67.16,67.03,67.32,67.80,68.35,69.56]
# y_07 = [51.69,51.85,55.46,56.07,56.99,57.74,57.99,58.10,57.88,57.90,58.40,58.72,59.19,60.24]
# cost = [0,4079,11504,28335,46583,61524,95363,141641,186484,236614,379190,510912,603691,955363]

# ss
# y_03 = [54.77,56.09,58.83,59.20,59.47,59.47,59.61,59.76,59.67,58.36,57.75,56.15]
# y_05 = [49.53,50.72,54.00,54.95,55.50,55.50,55.65,55.67,55.43,53.50,52.39,51.10]
# y_07 = [43.38,43.52,46.07,46.16,47.56,47.29,47.54,47.57,47.19,44.19,42.52,40.76]
# cost = [0,4079,11504,28335,61524,95363,141641,236614,379190,510921,603691,955363]

# ps
y_03 = [54.77,56.45,59.30,60.38,60.50,60.69,61.06,60.96,60.98,59.84,57.11,56.15]
y_05 = [49.53,50.94,55.01,56.22,56.60,56.84,57.40,57.16,57.12,55.55,52.21,51.10]
y_07 = [43.38,43.88,46.06,47.17,48.19,48.38,48.35,48.26,48.26,46.50,41.91,40.76]
cost = [0,4570,13567,32422,53742,72456,114633,170560,225051,433990,650351,955363]

plt.plot(cost, y_03, c='blue', label="mAP-0.3")
plt.scatter(cost, y_03, c='blue')
plt.plot(cost, y_05, c='green', label="mAP-0.5")
plt.scatter(cost, y_05, c='green')
plt.plot(cost, y_07, c='red', label="mAP-0.7")
plt.scatter(cost, y_07, c='red')

# p

# plt.axhline(y = 69.56, color = 'green', linestyle = '--',label="all fusion")
# plt.axhline(y = 72.17, color = 'blue', linestyle = '--',label="all fusion")
# plt.axhline(y = 60.24, color = 'red', linestyle = '--',label="all fusion")

# plt.axhline(y = 59.04, color = 'green', linestyle = '-.',label="no fusion")
# plt.axhline(y = 63.50, color = 'blue', linestyle = '-.',label="no fusion")
# plt.axhline(y = 51.69, color = 'red', linestyle = '-.',label="no fusion")
# plt.axhline(y = 68.28, color = 'blue', linestyle = '--',label="best")

# plt.scatter([955363], [69.56], c='green')
# plt.scatter([0], [59.04], c='green')

# s 
plt.axhline(y = 56.15, color = 'blue', linestyle = '--',label="all fusion")
plt.axhline(y = 51.10, color = 'green', linestyle = '--',label="all fusion")
plt.axhline(y = 40.76, color = 'red', linestyle = '--',label="all fusion")

# plt.axhline(y = 59.04, color = 'green', linestyle = '-.',label="no fusion")
# plt.axhline(y = 63.50, color = 'blue', linestyle = '-.',label="no fusion")
# plt.axhline(y = 51.69, color = 'red', linestyle = '-.',label="no fusion")
# plt.axhline(y = 68.28, color = 'blue', linestyle = '--',label="best")

plt.legend(loc='best')
plt.savefig('/workspace/ps_up_points.png')
