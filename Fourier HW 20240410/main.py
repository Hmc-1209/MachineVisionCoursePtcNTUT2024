import numpy as np

decimal = 2
f = np.array([1, 2, 4, 3, 2, 1, 1])
g = np.array([1, 1, 1, 1, 1, 1, 1]) * (1/7)

convolve_result = np.convolve(f, g, mode='full')
print("(a) 卷積的結果 f*g:")
print(np.around(convolve_result, decimal), '\n')

ft_result = np.fft.fft(convolve_result)
print("(b) 卷積的離散傅立葉轉換 F(f*g):")
print(np.around(ft_result, decimal), '\n')

# Calculate the padding
ftf = np.fft.fft(f, n=13)
ftg = np.fft.fft(g, n=13)

product_ft = ftf * ftg
print("(c) 分別求離散傅立葉轉換後，再求乘積 F(f)∙F(g):")
print(np.around(product_ft, decimal), '\n')

inverse_fft = np.fft.ifft(ft_result)
print("(e) 根據(b)的結果求反離散傅立葉轉換，並與(a)的結果比較:")
print(np.round(inverse_fft, decimal))
