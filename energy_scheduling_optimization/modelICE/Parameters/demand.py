import numpy as np
import matplotlib.pyplot as plt

# dayahead_ele_load_list = [822, 798, 803, 809, 851, 971, 1103, 1211, 1282, 1306, 1306,1300,
#      1287, 1281, 1263, 1245, 1214, 1214, 1130, 1064, 1009, 931, 901, 852]
# dayahead_ele_load_list = np.ones(24,dtype=np.int32)*5000
# dayahead_ele_load_list = [16920.81,17079.99,16962.69,17067.61,17257.38,18268.22,19507.87,19657.62,18190.22,17792.99,17765.4,18223.28,18466.42,19058.84,18948.44,18670.65,18295.79,18021.99,17725.7,17766.74,17669.39,17669.39,17669.39,17669.39
# ]
dayahead_ele_load_list = [ 796.4243,  796.7997,  788.4807,  782.4283,  788.1214,  817.6466,  856.4361,
  894.5286,  925.3066,  924.1897,  965.362,  1011.4984, 1059.9926, 1104.1652,
 1113.2532, 1106.9195, 1072.8737, 1064.2597,  926.171,   903.4022,  877.6817,
  836.4817,  829.2817,  822.8817]
dayahead_cold_load_list = [722, 711, 699, 676, 676, 674, 678, 762, 949, 976, 1081, 1162,
     1265, 1331, 1362, 1367, 1310, 1309, 986, 926, 869, 766, 748, 732]  # 空间冷负荷 + 500kw冷冻负荷

#[1200,1200,800,500,2000,400,400]
dayahead_ele_load = np.array(dayahead_ele_load_list)
dayahead_cold_load = np.array(dayahead_cold_load_list)
print("day ahead ele load:",dayahead_ele_load)
print("day ahead cold load:",dayahead_cold_load)
#
# M=24
# t_list =np.arange(0,M,1)
# plt.bar(t_list,dayahead_ele_load,align="center", color="r",label='dayahead_ele_load')
# plt.bar(t_list,dayahead_cold_load, align="center", bottom=dayahead_ele_load, color="y", label="elechiller_ele_load")
# # plt.bar(t_list,dayahead_cold_load_list, align="center", bottom=dayahead_ele_load+dayahead_cold_load, color="g", label="elechiller_ele_load_list")
# plt.legend()
# plt.xlabel("hour",fontsize=8)  # 设置x，y轴
# plt.ylabel("cold power",fontsize=8)
# plt.show()
# plt.subplot(212)
# plt.bar(t_list,dayahead_ele_load)