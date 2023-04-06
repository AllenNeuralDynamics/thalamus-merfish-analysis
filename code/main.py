try:
    import sys
    import matplotlib.pyplot as plt
    from numpy import linspace, pi, sin

    if (len(sys.argv) >= 4):
        plot_title = sys.argv[1]
        cycles = int(sys.argv[2])
        input_data = sys.argv[3]
    else:
        plot_title = "Hello Code Ocean"
        cycles = 3
        input_data = "../data/sample-data.txt"

    # read some input data from a file provided as an argument
    with open(input_data, 'r') as f:
        points = int(f.read())

    # plot the sine function
    x = linspace(0, cycles * 2 * pi, points)
    y = sin(x)

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot_title)

    # finally, save the resulting plot to a PNG file (note the output directory)
    plt.savefig('../results/fig1.png')

except ImportError:
    with open('../results/result.txt', 'w') as f:
        f.write('Hello, World!')
    print("To generate a result figure, please install `matplotlib` via conda. See README.md for more details.")
