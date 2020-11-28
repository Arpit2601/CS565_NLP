#include<bits/stdc++.h>
using namespace std;
// V, N
int vocab_size,  num_hidden_neurons;
long double learning_rate;

// Weight matrix is of size V*N and Weight dash is of size N*V
long double** Weight_matrix, **Weight_dash_matrix;

// hidden layer is of size N*1 and output layer is of size V*1
long double * hidden_layer, *output_layer;

void iterate(int iter_id, vector<int> training_sample)
{
    int input_word_id = training_sample[1], output_word_id = training_sample[2];
    
    //-----------FORWARD PASS---------------
    // Compute hidden layer values, which will be same as that of Weight[input_word_id]
    for (int i=0;i<num_hidden_neurons;i++)
    {
        hidden_layer[i] = Weight_matrix[input_word_id][i];
    }

    // summation exp(O[i])
    long double total_probablities = 0;

    // Compute the output layer values
    for(int i=0;i<vocab_size;i++)
    {
        output_layer[i] = 0;
        // O[i] += H[j] * W'[j][i] for j in hidden layer
        for(int j=0;j<num_hidden_neurons;j++)
        {
            output_layer[i] += hidden_layer[j] * Weight_dash_matrix[j][i];
        }
        total_probablities += (long double)exp(output_layer[i]);
    }

    // convert output layer values to probablities, applying softmax
    for (int i=0;i<vocab_size;i++)
    {
        output_layer[i] = (long double)(exp(output_layer[i]) / total_probablities);
    }

    //-----------BACKWARD PASS---------------

    // First calculate all the gradients  then update the weights
    //Hidden -> input updates
    // Total error for each hidden layer neuron based on all output vectors
    long double EH[num_hidden_neurons] = {0};
    for (int i=0;i<num_hidden_neurons;i++)
    {
        // EH[i] += W'[i][j] * error corresponding to jth neuron in output layer
        for(int j=0;j<vocab_size;j++)
        {
            EH[i] += Weight_dash_matrix[i][j] * (j == output_word_id ? output_layer[j] - 1 : output_layer[j]);
        }
    }


    // Output -> hidden updates
    // Since considering extreme case of negative sampling only update the column of W' corresponding to the output_word_id
    int positive_weight_update = 0, negative_weight_update = 0;

    // error corresponding to output layer neuron of output word id
    long double error = output_layer[output_word_id] - 1;

    // Applying updates to W' matrix
    for (int i=0;i<num_hidden_neurons;i++)
    {
        Weight_dash_matrix[i][output_word_id] -= learning_rate * error * hidden_layer[i];
        if (learning_rate * error * hidden_layer[i] > 0)
        {
            negative_weight_update ++;
        }
        else
        {
            positive_weight_update ++;
        }
        
    }

    
    // Applying updates to Weight matrix
    for(int i=0;i<num_hidden_neurons;i++)
    {
        Weight_matrix[input_word_id][i] -= learning_rate * EH[i];
        if (learning_rate * EH[i] > 0)
        {
            negative_weight_update ++ ;
        }
        else
        {
            positive_weight_update ++;
        }
        
    }

    cout<<iter_id + 1<<" "<<training_sample[0]<<" "<<negative_weight_update<<" "<<positive_weight_update<<endl;

}

int main() 
{
    // Taking vocab size (V), number of hidden layer neurons (N), number of iterations and number of training samples as input
    cin>>vocab_size;
    cin>>num_hidden_neurons;
    cin>>learning_rate;
    int num_iterations;
    cin>>num_iterations;
    int num_training_samples;
    cin>>num_training_samples;
    int id, word1, word2;
    vector<vector<int>> training_samples;

    // Initialising Weight matrix (W)
    Weight_matrix = (long double **)malloc(sizeof(long double) * vocab_size * num_hidden_neurons);
    for (int i=0;i<vocab_size;i++)
    {
        Weight_matrix[i] = (long double *)malloc(num_hidden_neurons * sizeof(long double));
        for (int j=0;j<num_hidden_neurons;j++)
        {
            Weight_matrix[i][j] = (long double) (0.5); 
        }
    }

    // Initialising Weight dash matrix (W')
    Weight_dash_matrix = (long double **)malloc(sizeof(long double) * vocab_size * num_hidden_neurons);
    for (int i=0;i<num_hidden_neurons;i++)
    {
        Weight_dash_matrix[i] = (long double* )malloc(vocab_size*sizeof(long double));
        for (int j=0;j<vocab_size;j++)
        {
            Weight_dash_matrix[i][j] = (long double) (0.5); 
        }
    }

    // Initialise hidden layer (H) with zeros, will change with each training sample in iteration
    hidden_layer = (long double *)malloc(sizeof(long double) * num_hidden_neurons);
    for(int i=0;i<num_hidden_neurons;i++)
    {
        hidden_layer[i]=0;
    }

    // Initialise output layer (O) with zeros, will change with each training sample in iteration
    output_layer = (long double *)malloc(sizeof(long double) * vocab_size);
    for(int i=0;i<vocab_size;i++)
    {
        output_layer[i]=0;
    }

    // taking training samples as input
    for (int i=0;i<num_training_samples;i++)
    {
        cin>>id>>word1>>word2;
        // word1, word2 as input are 1 based, whereas indexing in matrices in zero based
        training_samples.push_back({id, word1-1, word2-1});
    }

    // Call iterate for all the samples for each iteration
    for (int iter_id=0;iter_id<num_iterations;iter_id++)
    {
        for (int j=0;j<training_samples.size();j++)
        {
            iterate(iter_id, training_samples[j]);
        }
    }

}