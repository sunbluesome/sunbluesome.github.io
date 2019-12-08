---
layout: default
title: トップページ
---



# 余因子行列を用いる方法
余因子$\Delta_{ij}$は対象の行列の$i$行目と$j$列目を除いた行列の行列式を$(−1)i+j$倍したもの。
$$
\begin{align}
A^{-1} = \frac{\tilde A}{\det A}
\end{align}
$$
と表せる。

# 行列式の余因子展開
行列式は、任意の行または列で成分と余因子を用いて
$$
\begin{align}
\det A = \sum_{i} a_{ij} \Delta_{ij} = \sum_{j} a_{ij}\Delta_{ij}
\end{align}
$$
のように展開できる。

# 逆行列の求め方（余因子行列の添え字に注意）
掃き出し法というのもあるが、一般化のために余因子を用いる方法をメモしておく。  
　余因子を$i,j$成分とした余因子行列$\tilde A$を用いて、行列Aの逆行列$A^{-1}$は以下のように計算できる。
$$
\begin{align}
\left\{\begin{array}{l}
A^{-1} &= \frac{\tilde A}{|A|} \\
\tilde A &= \left|\begin{array}{llll}
    \Delta_{11} & \Delta_{12} & \ldots & \Delta_{1n} \\ 
    \Delta_{21} & \Delta_{22} & \ldots & \Delta_{2n} \\ 
    \vdots & \vdots & \ddots & \vdots \\ 
    \Delta_{n1} & \Delta_{n2} & \ldots & \Delta_{nn} \\ 
    \end{array}\right|
\end{array}\right.
\end{align}
$$

# 行列式の微分（余因子行列の添え字に注意）
行列式の余因子展開と、逆行列と行列式の関係より
$$
\begin{align}
\frac{\partial |A|}{\partial a_{ij}} = \Delta_{ji} = |A|b_{ji}
\end{align}
$$
が得られる。なお、行列Aの逆行列の成分を$b_{ij}$とした。これより、
$$
\begin{align}
\frac{\partial |A|}{\partial x} &= \frac{\partial |A|}{\partial a_{ij}} \frac{\partial a_{ij}}{\partial x} = |A|b_{ji}\frac{\partial a_{ij}}{\partial x} \\
&= |A| \mathrm{tr}\left(A^{-1}\frac{\partial A}{\partial x}\right)
\end{align}
$$

## 行列式の対数の微分
行列式の微分より、
$$
\begin{align}
\frac{\partial}{\partial x} \log |A| &= \frac{1}{|A|} \frac{\partial |A|}{\partial x} = \mathrm{tr} \left(A^{-1}\frac{\partial A}{\partial x}\right)
\end{align}
$$

# 逆行列の微分
$$
\begin{align}
AA^{-1} &= I \\
\end{align}
$$
これを両辺$x$で微分すると
$$
\begin{align}
\frac{\partial A^{-1}}{\partial x} A + A^{-1} \frac{\partial A}{\partial x} &= \boldsymbol{0} \\
\frac{\partial A^{-1}}{\partial x} &= A^{-1} \frac{\partial A}{\partial x} A^{-1} \\
\end{align}
$$
となる。


