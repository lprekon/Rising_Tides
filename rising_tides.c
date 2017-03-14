#include "lodepng.h"
#include "neuralNet.h"
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define OUTPUT_NODES 26
#define LIBRARY_SIZE 4
/* One training set is composed of 60 lessons. Each lesson consists of applying one 
   input for each character, averaging the gradient over each input, and applying it at
   the end of the lesson. Thus one training set modifies the weight matrices 60 times.*/
#define TRAINING_SET_SIZE 60
#define LEARNING_FACTOR 0.1  // Chosen mostly arbitarily.

static float *input_activation;
static struct neural_layer *hidden_layers;
static unsigned hidden_layer_count;
static struct neural_layer output_layer;
static struct image **image_library;  // REMEMBER SELF, THIS IS A MATRIX OF IMAGE STRUCTS
static char* id_string;  // Gene string, I just didn't want to reuse the name.

static void clean_finish();
static void clean_exit();


/*////////////////////////////////////////////////////////////////////
////  Warning: This program includes very little error checking!  ////
////          Make sure your data is properly formed.             ////
////////////////////////////////////////////////////////////////////*/


int main(int argc, char const *argv[])
{	
	/* Parse the gene string to get neural network parameters. 
		For now I'm making the (possibly dangerous) assumption
		that the string is properly formed. */
	#ifdef DEBUG_REPORT
	printf("[*] Starting in debug mode\n");
	#endif
	printf("[*] Beginning new neural network\n");
	unsigned string_length = strlen(argv[1]);
	char* gene_string = malloc(sizeof(char) * string_length);
	memcpy(gene_string, argv[1], string_length);
	id_string = malloc(sizeof(char) * string_length);
	memcpy(id_string, argv[1], string_length);
	unsigned parameter_count = 1;
	char* iterator = gene_string;
	while(*iterator != '\0')
	{
		if(*iterator == ':')
			parameter_count++;
		iterator++;
	}
	char **parameter_list = (char **)malloc(sizeof(char *) * parameter_count);
	iterator = gene_string;
	parameter_list[0] = gene_string;
	parameter_count = 1;
	while(*iterator != '\0')
	{
		if(*iterator == ':')
		{
			*iterator = '\0';
			parameter_list[parameter_count++] = iterator + 1;
		}
		iterator++;
	}
	/* Before we begin asigning resources, install clean-up call to ensure they'll be freed. */
	printf("[*] Assigning clean-up... " );
	atexit(clean_exit);
	printf("Done.\n");
	char *chp_input_k;
	iterator = parameter_list[0];
	printf("[*] Parsing for input paramters... ");
	while(*iterator != '\0')
	{
		if(*iterator == '-')
		{
			chp_input_k = iterator + 1;
			*iterator = '\0';
			break;
		}
		iterator++;
	}
	printf("Done.\n");
	unsigned input_size = (unsigned)atoi(parameter_list[0]);
	unsigned input_k = (unsigned)atoi(chp_input_k);
	input_activation = malloc(sizeof(float) * (input_size + 1));
	input_activation[input_size] = 1;
	/* Calculate best aspect ratio for input. i.e how should the image be divided and given to the input nodes. */
	unsigned start = (unsigned int)sqrt(input_size);
	int i, j;
	unsigned aspect_x, aspect_y;
	for(i = 0; i < input_size; i++)
	{
		if(input_size % (start - i) == 0)
		{
			aspect_x = start - (unsigned)i;
			aspect_y = input_size / aspect_x;
			break;
		}
		else if(input_size % (start + i) == 0)
		{
			aspect_x = start + (unsigned)i;
			aspect_y = start / aspect_x;
			break;
		}
	}
	printf("[+] Input layer created\n");
	printf("[+] Begin construction of hidden layers... ");
	/* Begin building hidden layers. */
	hidden_layer_count = parameter_count - 2;
	hidden_layers = malloc(hidden_layer_count * sizeof(struct neural_layer));
	unsigned size_prev = input_size;  // Number of nodes in previous layer
	unsigned layer = 0;
	while (layer < hidden_layer_count)
	{
		char* layer_size = parameter_list[layer + 1];
		iterator = layer_size;
		char* layer_k = layer_size;
		while(*iterator != '\0')
		{
			if(*iterator == '-')
			{
				layer_k = iterator + 1;
				*iterator = '\0';
				break;
			}
			iterator++;
		}
		hidden_layers[layer].k = atof(layer_k);
		hidden_layers[layer].height = atoi(layer_size);  // Number of nodes in this layer
		hidden_layers[layer].width = size_prev;
		hidden_layers[layer].weight = calloc(sizeof(float *), hidden_layers[layer].height);
		hidden_layers[layer].gradient = calloc(sizeof(float *), hidden_layers[layer].height);
		hidden_layers[layer].input = malloc(sizeof(float) * hidden_layers[layer].height);
		hidden_layers[layer].error = malloc(sizeof(float) * hidden_layers[layer].height);
		hidden_layers[layer].output = malloc(sizeof(float) * hidden_layers[layer].height + 1);
		hidden_layers[layer].output[hidden_layers[layer].height] = 1;
		for(i = 0; i < hidden_layers[layer].height; i++)
		{
			hidden_layers[layer].weight[i] = malloc(sizeof(float) * (size_prev + 1));  // +1 to account for bias node
			hidden_layers[layer].gradient[i] = malloc(sizeof(float) * (size_prev + 1));
			for(j =0; j <= size_prev; j++)  // <= equivalent to < +1 to account for bias variable.
			{
				hidden_layers[layer].weight[i][j] = getWeight();  // Insert the random weights as we build the matrix
			}
		}
		size_prev = hidden_layers[layer].height;
		layer++;
	}
	printf("Done.\n");
	/* Build the output layer */
	printf("[+] Building output layer... ");
	output_layer.k = atof(parameter_list[parameter_count - 1]);  // The very last element is the k value for output
	free(parameter_list);
	output_layer.height = OUTPUT_NODES;
	output_layer.width = size_prev + 1;  // Upon exit from the loop, size_prev is set to the size of the last hidden layer
	output_layer.input = malloc(sizeof(float) * output_layer.height);
	output_layer.error = malloc(sizeof(float) * output_layer.height);
	output_layer.output = malloc(sizeof(float) * output_layer.height);  // Here, output is the same size as input because there is no bias node
	output_layer.weight = calloc(sizeof(float *), output_layer.height);
	output_layer.gradient = calloc(sizeof(float *), output_layer.height);
	for(i = 0; i <output_layer.height; i++)
	{
		output_layer.weight[i] = malloc(sizeof(float *) * output_layer.width);
		output_layer.gradient[i] = malloc(sizeof(float *) * output_layer.width);
		for(j = 0; j <= size_prev; j++)
		{
			output_layer.weight[i][j] = getWeight();
		}
	}
	printf("Done.\n");
	printf("[+] Neural Network finished construction\n");
	/* Neural Network has finished construction. */
	/* Now load the image library. */
	#ifndef DEBUG_REPORT
	printf("[+] Load image library... ");
	image_library = calloc(sizeof(struct image), OUTPUT_NODES);
	char filepath[] = "A0.png";
	unsigned error;
	for(i = 0; i < OUTPUT_NODES; i++)
	{

		filepath[0] = (char)(65 + i); // sets the first character to A-Z.
		image_library[i] = malloc(sizeof(struct image) * LIBRARY_SIZE);
		for(j = 0; j < LIBRARY_SIZE; j++)
		{
			filepath[1] = (char)(48 + j); // sets the second character to 0-9. NOTE: This code does not support libraries of size > 10
			error = lodepng_decode_file(&image_library[i][j].pixel_data, \
				&image_library[i][j].width, &image_library[i][j].height, filepath, LCT_GREY, 8);
			if(error != 0)
			{
				printf("%s\n[-] ERROR! Part of the image library could not be loaded!\n Exiting...\n", id_string);
				exit(0);
			}
		}
	}
	#endif
	printf("Done.\n");
	/* Time to train the network. The first step is to calculate our current statistics
	   and report them. */
	test_and_report(0, input_k, aspect_x, aspect_y);
	#ifndef DEBUG_REPORT
	printf("[+] %s created, tested and reported. Beginning training...\n", id_string);
	int set_counter = 0;
	while(1)
	{
		run_training_set(input_k, aspect_x, aspect_y);
		set_counter += 1;
		test_and_report(set_counter, input_k, aspect_x, aspect_y);
	}
	#endif
}

void test_and_report(unsigned set_num, float input_k, unsigned aspect_x, unsigned aspect_y)
{
	float accuracy = 0.0, accept = 0.0, reject = 0.0, high_score = 0.0;
	int i, j, k;
	#ifndef DEBUG_REPORT
	for(i = 0; i < OUTPUT_NODES; i++)
	{
		for(j = 0; j < LIBRARY_SIZE; j++)
		{
			activate_network(&image_library[i][j], input_k, aspect_x, aspect_y);
			high_score = (high_score < output_layer.output[i]) ? output_layer.output[i] : high_score;
			unsigned guess = 0;
			for(k = 0; k < OUTPUT_NODES; k++)
			{
				guess = (output_layer.output[k] > output_layer.output[guess]) ? k : guess;
				if(k != i)
					reject += output_layer.output[k];
			}
			if(guess == i)
				accuracy += 1;  // We guessed the correct letter
			accept += output_layer.output[i];
		}
	}
	accuracy /= OUTPUT_NODES * LIBRARY_SIZE;
	accept /= OUTPUT_NODES * LIBRARY_SIZE;
	reject /= OUTPUT_NODES * LIBRARY_SIZE;
	#else
		accuracy = 1.0, accept = 1.1, reject = 1.12, high_score = 0.999;
	#endif

	/* Tests have been run, now assemble them into the report. */
	char POST_format[] = "POST /users/lprekon/report.cgi HTTP/1.1\r\nHost: cs.utexas.edu\r\nAccept: text/html, text/plain\r\nAccept-Language: en-us\r\n\r\nid=%s&sets=%.15d&avgAcc=%.15f&avgAcp=%.15f&avgRej=%.15f&high=%.15f";
	int url_length = strlen(POST_format) + strlen(id_string) + 15*5 + 1; // 15 for length of the metrics (set #, accuracy, etc.) time 5 metrics reported, plus 1 for the null byte at the end.
	char *post_request = malloc(sizeof(char) * url_length);
	snprintf(post_request, url_length, POST_format, id_string, set_num, accuracy, accept, reject, high_score);
	#ifdef DEBUG_REPORT
		printf(" Will send %s\n\n", post_request);
	#endif
	
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0)
	{
		printf("Socket could not be created\n");
		return;
	}
	struct sockaddr_in serv_addr;
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(80);
	const char serv_ip[] = "128.83.120.11";
	serv_addr.sin_addr.s_addr = inet_addr(serv_ip);
	if(connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
		printf("Could not connect\n");
		return;
	}

	send(sockfd, post_request, strlen(post_request), 0);
	#ifdef DEBUG_REPORT
		char buffer[100];
		while(recv(sockfd, buffer, 100, 0) > 0)
			printf("%s", buffer);
	#else
		char buffer[] = {'A', 'A', 'A', '\0'};
		printf("You forgot to implement this dummy\n");
		return;
	#endif
	int command = atoi(buffer);
	if(command == 0)
		return;

	/* If we reach this point, then C&C has commanded that this trainer switch to a new neural network. */

	char *new_string = malloc(sizeof(char) * command);
	recv(sockfd, new_string, command, 0);
	printf("[*] %s switching to %s\n", id_string, new_string);
	char *argv[] = {"neuralNet", new_string, (char *)NULL};
	execv(argv[1], argv);
	printf("[*] ERROR! %s HAS FAILED TO SWITCH.\n", id_string);
}

void run_training_set(float input_k, unsigned aspect_x, unsigned aspect_y)
{
	int i;
	for(i = 0; i < TRAINING_SET_SIZE; i++)  // Each iteration is one lesson
	{
		/* Reset all gradient matrices to 0*/
		int l, w, h;
		for(l = 0; l < hidden_layer_count; l++)
		{
			for(h = 0; h < hidden_layers[l].height; h++)
			{
				for(w = 0; w < hidden_layers[l].width; w++)
				{
					hidden_layers[l].gradient[h][w] = 0.0;
				}
			}
		}
		for(h = 0; h < output_layer.height; h++)
		{
			for(w = 0; w < output_layer.width; w++)
			{
				output_layer.gradient[h][w] = 0.0;
			}
		}
		int letter;
		for(letter = 0; letter < OUTPUT_NODES; letter++)
		{
			activate_network(&image_library[letter][i % LIBRARY_SIZE], input_k, aspect_x, aspect_y);
			backpropogate_error(letter);
			/* Now to calculate the gradients. They'll all be added together then divided at the end 
			   to get the average. */
			float *prev_layer_activation = input_activation;
			for(l = 0; l < hidden_layer_count; l++)
			{
				for(h = 0; h < hidden_layers[l].height; h++)
				{
					for(w = 0; w < hidden_layers[l].width; w++)
					{
						hidden_layers[l].gradient[h][w] += hidden_layers[l].error[h] * prev_layer_activation[w];
					}
				}
				prev_layer_activation = hidden_layers[l].output;
			}
			for(h = 0; h < output_layer.height; h++)
			{
				for(w = 0; w < output_layer.width; w++)
				{
					output_layer.gradient[h][w] += output_layer.error[h] * hidden_layers[hidden_layer_count - 1].output[w];
				}
			}
		}
		/* OUTPUT_NODES number of inputs applied, gradients summed up. Now to divide
		   to get the average, multiply it by the learning factor, and update the weight
		   matrices. */
		for(l = 0; l < hidden_layer_count; l++)
		{
			for(h = 0; h < hidden_layers[l].height; h++)
			{
				for(w = 0; w < hidden_layers[l].width; w++)
				{
					hidden_layers[l].weight[h][w] -= hidden_layers[l].gradient[h][w] / OUTPUT_NODES * LEARNING_FACTOR;
				}
			}
		}
		for(h = 0; h < output_layer.height; h++)
		{
			for(w = 0; w < output_layer.width; w++)
			{
				output_layer.weight[h][w] -= (output_layer.gradient[h][w] / OUTPUT_NODES * LEARNING_FACTOR);
			}
		}
		/* Lesson complete. */
	}
}

void activate_network(struct image *input_data, float input_k, unsigned aspect_x, unsigned aspect_y)
{
	unsigned input_layer_size = hidden_layers[0].width - 1;
	/* Step 1: distribute data to the input vector*/
	unsigned w = input_data->width;
	unsigned h = input_data->height;
	char* data = input_data->pixel_data;
	if(w % aspect_x != 0 || h % aspect_y != 0)
		printf("[*] Warning: aspect ratio does not conform to input dimensions. Expect depreciated accuracy.\n");
	unsigned chunk_x = w / aspect_x;
	unsigned chunk_y = h/aspect_y;
	unsigned pixels_per_node = chunk_x * chunk_y;
	int i, x, y;
	for(i = 0; i < input_layer_size; i++)
	{
		unsigned start_y = i / aspect_x * chunk_y;
		unsigned start_x = i % aspect_x * chunk_x;
		float avg_color = 0;
		for(y = start_y; y < start_y + chunk_y; y++)
		{
			for(x = start_x; x < start_x + chunk_x; x++)
			{
				avg_color += data[y*w + x];
			}
		}
		avg_color /= pixels_per_node;
		avg_color -= 177;  // 255 (the maximum pixel value) divided by 2, to make the data more meaningful to sigmoid function
		input_activation[i] = (float)(1/(1 + exp(-avg_color * input_k)));
	}
	/* Now to propogate the values through hidden layers. We can infer from the current layer the size of the previous. */
	float *activation = input_activation;
	unsigned l;
	for(l = 0; l < hidden_layer_count; l++)
	{
		float *layer_input = hidden_layers[l].input;
		float *layer_output = hidden_layers[l].output;
		float k_value = hidden_layers[l].k;
		unsigned height = hidden_layers[l].height;
		move_data(activation, layer_input, hidden_layers[l].weight, hidden_layers[l].width, height);
		int h;
		for(h = 0; h < height; h++)
		{
			layer_output[h] = (float)(1/(1 + exp(-layer_input[h] * k_value)));
		}
		activation = hidden_layers[l].output;
	}
	/* Finished with the hidden layers, now just for the output layer. */
	move_data(activation, output_layer.input, output_layer.weight, output_layer.width, output_layer.height);
	for(i = 0; i <output_layer.height; i++)
	{
		output_layer.output[i] = (float)(1/(1 + exp(-output_layer.input[i] * output_layer.k)));
	}
	/* And we're done! The neural network has been successfully activated*/
}

void backpropogate_error(int intended_character)
{
	/* output_error = (actual - expected) * sigmoid'(input) 
	   sigmoid'(x) = k * e^kx/(1 + e^kx)^2 */
	int i;
	for(i = 0; i < output_layer.height; i++)
	{
		int expected = (i != intended_character)? 0 : 1;
		float sigmoid_prime = output_layer.k * exp(output_layer.input[i]) / pow(exp(output_layer.input[i]) + 1, 2);  
		output_layer.error[i] = (output_layer.output[i] - expected) * sigmoid_prime;
	}
	/* l_error_vector= (l+1_weights_transposed * l+1_error_vector) component_multiply simgoid'(l_input_vector) */
	struct neural_layer *following_layer = &output_layer;  // I'm moving backward through the layers. There's no easy way to name this.
	int l;
	for(l = hidden_layer_count - 1; l >= 0; l--)
	{
		move_data_transpose(following_layer->error, hidden_layers[l].error, following_layer->weight, \
			following_layer->width, following_layer->height);
		for(i = 0; i < hidden_layers[l].height; i++)
		{
			float k = hidden_layers[l].k;
			float x = hidden_layers[l].input[i];
			hidden_layers[l].error[i] = hidden_layers[l].error[i] * k * exp(k * x)/pow(exp(k * x) + 1, 2);
		}
		following_layer = &hidden_layers[l];
	}
	/* Done. All layers have their respective errors. Gradient calculation will be done elsewhere. */
}

/* Matrix-multiplies the n by 1 matrix 'from' with the m by n matrix 'weight', 
   storing the result in the m by 1 matrix 'to'. */
void move_data(float from[], float to[], float *weights[], unsigned width, unsigned height)
{
	/* Width and height refer to the weight matrix. From there the other necessary values can be infered. */
	int x, y;
	for(y = 0; y < height;  y++)
	{
		for(x = 0; x < width; x++)
		{
			to[y] += weights[y][x] * from[x];
		}
	}
}

/* A version of move_data used for backpropogation, and as such, implicitly transposes the weight matrix during multiplication. 
   Before, width was the length of 'from' and height was the length of 'to'. This is now reversed. 
   Optimized to still take advantage of row-major storage. */
void move_data_transpose(float from[], float to[], float *weights[], unsigned width, unsigned height)
{
	int y, x;
	for(y = 0; y < height; y++)
	{
		for(x = 0; x < width; x++)
		{
			to[x] += weights[y][x] * from[y];
		}
	}
}

float getWeight()
{
	return rand()/1000000000.0 - 1;
}


/* Make sure that our resources are released before we exit. */
void clean_finish()
{
	/* Because of clean_finish, some arrays of arrays are initialized with calloc, to guarantee
	   their entries are 0, and wont get accidentally freed if they are never assigned. */
	if(id_string)
	{
		printf("[*] %s clean finish\n", id_string);
		free(id_string);
	}
	else
		printf("[*]   clean_finish\n");
	if(input_activation)
		free(input_activation);
	int i, j;
	for(i = 0; i < hidden_layer_count; i++)
	{
		if(hidden_layers[i].input)
			free(hidden_layers[i].input);
		if(hidden_layers[i].output)
			free(hidden_layers[i].output);
		if(hidden_layers[i].error)
			free(hidden_layers[i].error);
		if(hidden_layers[i].weight)
		{
			for(j = 0; j < hidden_layers[i].height; j++)
			{
				if(hidden_layers[i].weight[j])
					free(hidden_layers[i].weight[j]);
			}
			free(hidden_layers[i].weight);
		}
		if(hidden_layers[i].gradient)
		{
			for(j = 0; j < hidden_layers[i].height; j++)
			{
				if(hidden_layers[i].gradient[j])
					free(hidden_layers[i].gradient[j]);
			}
			free(hidden_layers[i].gradient);
		}
	}
	if(output_layer.input)
		free(output_layer.input);
	if(output_layer.output)
		free(output_layer.output);
	if(output_layer.error)
		free(output_layer.error);
	if(output_layer.weight)
	{
		for(j = 0; j < output_layer.height; j++)
		{
			if(output_layer.weight[j])
				free(output_layer.weight[j]);
		}
		free(output_layer.weight);
	}
	if(output_layer.gradient)
	{
		for(j = 0; j < output_layer.height; j++)
		{
			if(output_layer.gradient[j])
				free(output_layer.gradient[j]);
		}
		free(output_layer.gradient);
	}
	if(image_library)
	{
		for(i = 0; i < OUTPUT_NODES; i++)
		{
			if(image_library[i])
			{
				for(j = 0; j < LIBRARY_SIZE; j++)
				{
					if(image_library[i][j].pixel_data)
						free(image_library[i][j].pixel_data);
				}
				free(image_library[i]);
			}
		}
		free(image_library);
	}
}

void clean_exit()
{
	clean_finish();
	exit(0);
}