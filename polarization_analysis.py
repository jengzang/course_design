import math

# 输入光强
IH = 210  # 水平
IV = 128  # 垂直
I45 = 107  # 45度
IR = 74  # 右旋圆偏振

# 计算斯托克斯参数
S0 = IH + IV
S1 = IH - IV
S2 = I45 - (S0 / 2)
S3 = IR - (S0 / 2)

# 输出斯托克斯参数
print("S0 =", S0)
print("S1 =", S1)
print("S2 =", S2)
print("S3 =", S3)

# 计算偏振度
P = math.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / S0
print("偏振度：", P)

# 判断偏振光类型
if S1 == 0 and S2 == 0 and S3 == 0:
    print("非偏振光")
elif S1 == 0 and S2 == 0:
    if S3 > 0:
        print("右旋圆偏振光")
    else:
        print("左旋圆偏振光")
else:
    if S3 > 0:
        print("右旋椭圆偏振光")
    elif S3 < 0:
        print("左旋椭圆偏振光")

    theta_lin = 0.5 * math.atan2(S2, S1)
    print("偏振角：", theta_lin * 180 / math.pi, "度")

    tuo = 0.5 * math.asin(S3 / S0)
    print("椭圆率：", tuo)
