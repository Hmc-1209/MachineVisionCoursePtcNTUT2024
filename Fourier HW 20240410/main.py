import numpy as np


def dft(arr):
    # Do Fourier Transform
    length = len(arr)
    ft_arr = []
    for k in range(length):
        sum_value = 0
        for n in range(length):
            sum_value += arr[n] * np.exp(-2j * np.pi * k * n / length)
        ft_arr.append(sum_value)
    return np.array(ft_arr)


decimal = 2
x = np.array([1, 2, 4, 3, 2, 1, 1])
g = np.array([1, 1, 1, 1, 1, 1, 1]) * (1/7)
ftx = dft(x)
ftg = dft(g)
print('x的傅立葉轉換：')
print(np.around(ftx, decimal), '\n')
print('g的傅立葉轉換：')
print(np.around(ftg, decimal), '\n')

convolve_result = np.convolve(x, g, mode='full')
print("(a) 卷積的結果 x*g:")
print(np.around(convolve_result, decimal), '\n')

ft_result = dft(convolve_result)
print("(b) 卷積的離散傅立葉轉換 F(x*g):")
print(np.around(ft_result, decimal), '\n')

product_ft = ftx * ftg
print("(c) 分別求離散傅立葉轉換後，再求乘積 F(x)∙F(g):")
print(np.around(product_ft, decimal), '\n')
