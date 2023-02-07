import matplotlib.pyplot as plt


plt.figure(figsize=(20, 10), dpi=100)
y_pp = [62.50,65.31,66.87,67.38,67.84,68.33,68.29,68.52,68.64,68.73]
x_pp = [12946.80,42317,73088,112148,154918,193681,241272,299966,357191,412541]
# y_negative = [59.55,64.04,64.22,68.02,68.99,69.19,69.33]
# x_negative = [10599,36633,59758,174722,245263,321401,443917]

y_ss = [51.86,52.96,55.28,55.28,55.47,55.73,55.81,55.64,55.38,54.36,53.73]
x_ss = [8483,38387,73392,112760,154136,192660,234559,286779,344278,403690,449994]
# yn_second = [49.58,49.72,50.25,52.06,52.59,53.22,54.66,55.38,54.99,54.82,54.33]
# xn_second = [4639,8773,14449,42100,66750,125854,188686,256824,323556,380658,435727]

y_sp = [62.01,64.15,65.69,66.46,66.81,66.79,67.01,67.13,67.56,67.49,67.75]
x_sp = [8483,38387,73392,112760,154136,192660,234559,286779,344278,403690,449994]
# yn_sp = [59.83,62.21,62.61,65.10,66.13,66.68,66.88,67.68]
# xn_sp = [14449,42100,66750,125854,188686,256824,323556,435727]

# plt.plot(x_pp, y_pp, c='red', label="pointpillars-pointpillars")
# # plt.plot(x_negative, y_negative, c='red', linestyle='--', label="negative pointpillars")
# plt.scatter(x_pp, y_pp, c='red')
# # plt.scatter(x_negative, y_negative,c='red')
# plt.axhline(y = 69.56, color = 'red', linestyle = '--',label="base")
# plt.scatter([955363.60], [69.56], c='red')

# plt.plot(x_ss, y_ss, c='green', label="second-second")
# # plt.plot(xn_second, yn_second, c='green', linestyle='--', label="negative second")
# plt.scatter(x_ss, y_ss, c='green')
# # plt.scatter(xn_second, yn_second,c='green')
# plt.axhline(y = 51.1, color = 'green', linestyle = '--')
# plt.scatter([955363.60], [51.1], c='green')

plt.plot(x_sp, y_sp, c='blue', label="second-pointpillars")
# plt.plot(xn_sp, yn_sp, c='blue', linestyle='--', label="negative sp")
plt.scatter(x_sp, y_sp, c='blue')
# plt.scatter(xn_sp, yn_sp,c='blue')
plt.axhline(y = 66.91, color = 'blue', linestyle = '--')
plt.scatter([58359.20], [66.91], c='blue')

plt.legend(loc='best')
plt.savefig('/workspace/ss.png')
