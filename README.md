# MedianFilter.jl

This is a simple median filter that resambles matlabs medfilt1. It is optimized for large windows by utilizing a heap based median calculation.

## Usage
The function medfilt1 accepts any Array containing real valued numbers (<:Real) and applies the median filter along the first non singelton dimension.
```julia
using MedianFilter

x = randn(Float64, 20)
y = medfilt1(x)
```

## Keyword arguments
Keyword `n` defines the window length. Default length is 3.
```julia
x = randn(Float64, 20)
y = medfilt1(x, n = 5)
```

Keyword `padding` defines the handling of endpoint. Default padding is `zeropad`, which results in all values outside the endpoints to be assumed as 0. Second variant `truncate` results in a reduction of window length when approaching the endpints.
```julia
x = randn(Float64, 20)
y = medfilt1(x, padding = "truncate")
```

Keyword `dim` defines the dimension along which the filter is applied. If not specified the first non singelton dimension gets selected.
```julia
x = randn(Float64, (20, 20))
y = medfilt1(x, dim = 2)
```

## Not supported
The Matlab keyword `nanflag` and therefore handling of missing values is not yet supported.