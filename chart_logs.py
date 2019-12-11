import argparse
from collections import Counter
from matplotlib import pyplot
import numpy as np
import re

def parse_args():
	parser = argparse.ArgumentParser("Extract counts for endings of episodes from logs")
	parser.add_argument("--log_file")
	parser.add_argument("--title")

	return vars(parser.parse_args())


def main():
	args = parse_args()

	log_file = args['log_file']
	title = args['title']
	counts = []
	rewards = []
	ind = 0
	counter = Counter()
	count_arr = [0,0,0,0,0]
	num_line = 0

	with open(log_file, 'r') as _file:
		# line = _file.readline()
		for line in _file:
			if line != "" and line[0:4] == "INFO":
				num_line += 1
				if (line[10:17] == 'Episode'):
					# ind = int(re.search(r'\d+', line).group())
					ind += 1
					start_of_counter_ind = line.find('C')
					counter = eval(line[start_of_counter_ind:-1])
					max_ind = 0
					max_count = 0
					count = 0
					for i in range(5):
						count = counter[i+1]
						if count > max_count:
							max_count = count
							max_ind = i

					count_arr[max_ind] += 1
					if (ind == 100):
						counts.append(count_arr)
						count_arr = [0,0,0,0,0]
						ind = 0

				elif (line[10] == '['):
					rewards.append(eval(line[10:-1]))

	# print(num_line)
	# print(counts)

	counts = np.array(counts).T

	# x = list(range(int(ind+1)))
	# print(x, ind)
	x = list(range(len(counts[0])))
	# print(len(x), counts.shape)


	# TODO: Make chart of rewards and ending scenarios
	# Q_BACK_FIRST_DOWN_LINE = 1
	# AGENT_OUT_OF_BOUNDS = 2
	# D_LINE_REACHED_Q_BACK = 3
	# Q_BACK_NOT_IN_BOUNDS = 4
	# Q_BACK_THREW_BALL = 5

	pyplot.plot(x, counts[0,:], 'b', label='Q_back first down')
	pyplot.plot(x, counts[1,:], 'g', label='Agent out of bounds')
	pyplot.plot(x, counts[2,:], 'r', label='D_line reached Q_b')
	pyplot.plot(x, counts[3,:], 'm', label='Q_back not in bounds')
	pyplot.plot(x, counts[4,:], 'c', label='Q_back threw ball')

	pyplot.legend()
	pyplot.xlabel("Episode (10^2)")
	pyplot.ylabel("Counts")
	pyplot.title(title)
# 
	pyplot.show()


if __name__ == "__main__":
	main()