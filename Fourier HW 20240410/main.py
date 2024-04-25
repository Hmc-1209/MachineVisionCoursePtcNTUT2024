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
f = np.array([1, 2, 4, 3, 2, 1, 1])
g = np.array([1, 1, 1, 1, 1, 1, 1]) * (1/7)
ftf = dft(f)
ftg = dft(g)
print('f的傅立葉轉換：')
print(np.around(ftf, decimal), '\n')
print('g的傅立葉轉換：')
print(np.around(ftg, decimal), '\n')

convolve_result = np.convolve(f, g, mode='full')
print("(a) 卷積的結果 f*g:")
print(np.around(convolve_result, decimal), '\n')

ft_result = dft(convolve_result)
print("(b) 卷積的離散傅立葉轉換 F(f*g):")
print(np.around(ft_result, decimal), '\n')

product_ft = ftf * ftg
print("(c) 分別求離散傅立葉轉換後，再求乘積 F(f)∙F(g):")
print(np.around(product_ft, decimal), '\n')

inverse_fft = np.fft.ifft(ft_result)
print("(e) 根據(b)的結果求反離散傅立葉轉換，並與(a)的結果比較:")
print(np.round(inverse_fft, decimal))
