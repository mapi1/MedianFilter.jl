using MedianFilter, Test

function tests()
    # all results were created using the Matlab medfitl1 function
    @testset "General Tests" begin
        @test medfilt1(collect(1:10), n = 3) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0]
    end

    @testset "Test Keyword n" begin
        @test_throws  AssertionError medfilt1(collect(1:10), n = -1)
    end

    @testset "Test Keyword padding" begin
        @test_throws AssertionError medfilt1(collect(1:10), padding = "none")
    end

    @testset "Test Keyword padding = 'zeropad'" begin
        @test medfilt1(collect(1:10), n=5, padding = "zeropad") == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]
        @test medfilt1(collect(1:10), n=6, padding = "zeropad") == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 7.5, 7.5]
    end

    @testset "Test Keyword padding = 'truncate'" begin
        @test medfilt1(collect(1:10), n=5, padding = "truncate") == [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0]
        @test medfilt1(collect(1:10), n=6, padding = "truncate") == [2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5, 8.0, 8.5]
    end


    @testset "Test Keyword dim" begin
        @test_throws AssertionError medfilt1(collect(1:10), dims = -3)
        @test_throws AssertionError medfilt1(collect(1:10), dims = -2)
        @test medfilt1([collect(1:5) collect(1:5)], n = 3) == [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 4.0 4.0]
        @test medfilt1([collect(1:5) collect(1:5)], n = 3, dims = 1) == [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 4.0 4.0]
        @test medfilt1([collect(1:5) collect(1:5)], n = 3, dims = 2) == [1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0; 5.0 5.0]
    end

end

tests()
