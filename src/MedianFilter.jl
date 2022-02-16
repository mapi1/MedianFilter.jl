# A Julia Implementation of a streaming MedianFilter using Heaps with the same syntax as Matlabs medfilt1
module MedianFilter
# Dependencies
using DataStructures
import Base.isless, Base.isequal

export medfilt1


"""
    medfilt1(x::Array{<:Real}; n::Int = 1, padding::String = "zeropad", dim::Int = -1)

Apply a median filter to a signal vector x, similar to Matlabs `medfilt1`. Using a heap-based calculation of the median to increase performance for larger windows n.

# Args:

* `x`: Input

#  Keywords:

* `n::Int`: Window length. The Median at point i is defined as median(x[i-n+1:i]), defaults to 1 returning `x`
* `padding::String`: Specifies how to deal with Endpoints. The modes 'zeropad' and 'truncate' are available, with the first as default. The mode `zeropad` pads zeros, so the window length is always `n`, while with `truncate` n goes to 1 at the endpoints. 
* `dim::Int`: Specifies the dimension to be filtered along. As default the first non singleton dimension is chosen.


# Return:

* Always returns an Array of the same dimensions as the input. If the input is a subtype of AbstractFloat the type is conserved. For other types like Int, the return will always be Float64.

# Examples

```jl
julia> medfilt1(1:10)
1:10

julia> medfilt1(collect(1:10))
10-element Vector{Int}:
 1
 2
 3
 4
 5
 6
 7
 8
 9
 10
```
"""
function medfilt1(x; n::Int = 1, padding::String = "zeropad", dims::Int = -1)
    #Check Input
    @assert n > 0 "The window n needs to be greater 0. Is: n = $n"
    @assert padding in ["truncate", "zeropad"] "$padding is not a valid mode for padding. Use 'truncate' or 'zeropad'."
    @assert maximum(size(x)) > 1 "There is no non singleton dimension in the input."
    if dims != -1
        @assert dims > 0 "The dimension needs to be positive. Currently: dims = $dims"
        @assert length(size(x)) >= dims "The input has only $(length(size(x))) dimensions but dims = $dims."
    else
        dims = findfirst(dim -> dim > 1, size(x))
    end

    if n == 1
        return x
    elseif padding == "zeropad"
        return mapslices(input -> medfilt1_worker_zeropad(input, n), x, dims = [dims])
    else
        return mapslices(input -> medfilt1_worker_truncate(input, n), x, dims = [dims])
    end

end

# The working horse, median Filter of an Vector with zero padding
function medfilt1_worker_zeropad(x, n::Int)
    if length(x) == 1
        return x
    end
    # adding zeros for padding
    x_pad = zeros(eltype(x), (n -1))
    x_iter = Base.Iterators.Stateful([x_pad; x; x_pad])

    # initialization phase
    (result, left, right, delete_queue) = initialize_filter(x_iter)
    currentMedian = result[2]
    # loop through all elements
    for i in x_iter
        # delete Element if n elements are in the delete_queue
        if length(delete_queue) == n
            delete_element = dequeue!(delete_queue)
            if delete_element.in_left
                delete!(left, delete_element.handle)
            else
                delete!(right, delete_element.handle)
            end
        end

        # add new item
        # compare to current median
        currentElement = MedianElement(i, false, -1)
        if currentElement.value >= currentMedian
            # add to left
            currentElement.handle = push!(right, currentElement)
        else
            # add to right
            currentElement.in_left = true
            currentElement.handle = push!(left, currentElement)
        end
        enqueue!(delete_queue, currentElement)

        # correct heaps if length differs more than one and get current median
        currentMedian = correct_heaps_return_median!(left, right)
        push!(result, currentMedian);
    end


    # cut result to the right length
    start = n + Int(ceil(n / 2)) - 1
    return result[start:start + length(x)-1]
end

# The working horse, median Filter of an Vector with smaller windows n at the endpoints
function medfilt1_worker_truncate(x, n::Int)
    if length(x) == 1
        return x
    end
    x_iter = Base.Iterators.Stateful(x)

    # initialization phase
    (result, left, right, delete_queue) = initialize_filter(x_iter)
    currentMedian = result[2]
    # loop through all elements an expand window (n_actual) to desired n
    n_actual = 2
    while length(delete_queue) > 1
        # delete Element if desired window size n was reached
        if n_actual >= n
            delete_element = dequeue!(delete_queue)
            if delete_element.in_left
                delete!(left, delete_element.handle)
            else
                delete!(right, delete_element.handle)
            end
        end

        # for each repetition increase n_actual
        n_actual += 1
        # add new item if there is something left in the iterator
        # compare to current median
        if !isempty(x_iter)
            currentElement = MedianElement(popfirst!(x_iter), false, -1)
            if currentElement.value >= currentMedian
                # add to left
                currentElement.handle = push!(right, currentElement)
            else
                # add to right
                currentElement.in_left = true
                currentElement.handle = push!(left, currentElement)
            end
            enqueue!(delete_queue, currentElement)
        end

        # correct heaps if length differs more than one and get current median
        currentMedian = correct_heaps_return_median!(left, right)
        push!(result, currentMedian);
    end

    # return desired excerpt
    start = Int(ceil((n) / 2))
    return result[start:start + length(x) - 1]
end

# Helper function that takes to Heaps, corrects them if length differs by more than 2 and returns the current median
function correct_heaps_return_median!(left, right)
    len_left = length(left)
    len_right = length(right)
    if abs(len_left - len_right) >= 2
        len_left > len_right ? transferTop!(left, right) : transferTop!(right, left)
        if abs(len_left - len_right) == 2
            currentMedian = 0.5 * first(left).value + 0.5 * first(right).value
            return currentMedian
        end
    end
    if len_left == len_right
        currentMedian = 0.5 * first(left).value + 0.5 * first(right).value
    elseif len_left > len_right
        currentMedian = first(left).value
    else
        currentMedian = first(right).value
    end
    return currentMedian
end

# Helper function that initializes all data structures and add first 2 elements
function initialize_filter(x_iter)
    # result vector
    if eltype(x_iter) <: AbstractFloat
        result = Vector{eltype(x_iter)}()
    else
        result = Vector{Float64}()
    end
    # >= current Median
    left = MutableBinaryMaxHeap{MedianElement}()
    # < current Median
    right = MutableBinaryMinHeap{MedianElement}()
    # A Queue that saves the Elements in the Heaps and pops the ones to delete next
    delete_queue = Queue{MedianElement}()

    # initialization phase
    # first value
    currentMedian = popfirst!(x_iter)
    push!(result, currentMedian);
    currentElement = MedianElement(currentMedian, true, -1)
    currentElement.handle  = push!(left, currentElement)
    enqueue!(delete_queue, currentElement)

    # second value
    currentElement = MedianElement(popfirst!(x_iter), false, -1)
    currentElement.handle = push!(right, currentElement)
    enqueue!(delete_queue, currentElement)
    if first(right).value < first(left).value
        swapTops!(left, right)
    end
    currentMedian = 0.5 * first(left).value + 0.5 * first(right).value
    push!(result, currentMedian);

    return (result, left, right, delete_queue)
end

# Data structure to store values and later identify them in a Heap to delete them
mutable struct MedianElement
    value
    in_left::Bool
    handle::Int
end

### Helper functions to get the Heap running on custom type MedianElement
function isless(x::MedianElement, y::MedianElement)
    return x.value < y.value
end
function isequal(x::MedianElement, y::MedianElement)
    return x.value == y.value
end
###

# Helper function to swap tops of the Heap
function swapTops!(left, right)
    temp_left = pop!(left)
    temp_left.in_left = false
    temp_right = pop!(right)
    temp_right.in_left = true
    temp_right.handle = push!(left, temp_right)
    temp_left.handle = push!(right, temp_left)
end

# Helper function to transfer the top to the smaller Heap
function transferTop!(larger, smaller)
    temp = pop!(larger)
    temp.in_left = !temp.in_left
    temp.handle = push!(smaller, temp)
end

end  # module MedianFilter
