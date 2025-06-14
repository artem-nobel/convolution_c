//
//  main.c
//  Deep Learning kernel finish
//  
//  Created by Артем Хорьков on 23.09.2023.
//
#includec <std
#include <stdio.h>
#include <math.h>

#define kernel_rows  3
#define kernel_cols  3

//double softmax(double x1,double x2)
//{
//    return x1,x2;
//}
printf("error 1");
double relu(double x)
{
    if(x < 0)
    {
        return 0;
    }
    return x;
}
int main(int argc, const char * argv[]) {
 double lr = 0.001;
    double lr_kernel = 0.01;
    int size_in = 5;
    int input[size_in][size_in];
    int goal_pred[] = {1,0,1,0,1};
    int goal_pred2[] = {0,1,0,1,0};
    int learning_data[5][5][5] = {
           {{1,1,1,1,1},
            {1,1,1,1,1},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0}},
        
           {{0,1,1,0,0},
            {0,1,1,0,0},
            {0,1,1,0,0},
            {0,1,1,0,0},
            {0,1,1,0,0}},
        
        {{0,0,0,0,0},
         {1,1,1,1,1},
         {1,1,1,1,1},
         {0,0,0,0,0},
         {0,0,0,0,0}},
        
        {{0,0,1,1,0},
         {0,0,1,1,0},
         {0,0,1,1,0},
         {0,0,1,1,0},
         {0,0,1,1,0}},
        
        {{0,0,0,0,0},
         {0,0,0,0,0},
         {0,0,0,0,0},
         {1,1,1,1,1},
         {1,1,1,1,1}}};
    
//    double kernel_weight[kernel_rows][kernel_cols] = {{2.907520, -2.582853, 2.907772},
//                                                      {0.790565, -4.754857, 0.790833},
//                                                      {2.887463, -2.536535, 2.887717}};
    double kernel_weight[kernel_rows][kernel_cols] = {{0.1,0.1,0.1},
                                                      {0.1,0.1,0.1},
                                                      {0.1,0.1,0.1}};

    double layer_1[3][3];
    
    double layer_1_kern_sum = 0;
    double layer2 = 0;
    double layer2_2 = 0;
    
    double layer1_2_weight[3][3] = {{0.1,0.1,0.1},
                                    {0.1,0.1,0.1},
                                    {0.1,0.1,0.1}};
    
    double layer1_2_weight2[3][3] = {{0.1,0.1,0.1},
                                    {0.1,0.1,0.1},
                                    {0.1,0.1,0.1}};
    
    double error =0;
    double error2 =0;
    
    
    for (int era_of_learning = 0; era_of_learning <100; era_of_learning ++)
    {
        for (int lear = 0; lear < 5; lear ++)
        {
            layer2 = 0;
            layer2_2 = 0;
            
            for (int n = 0; n < ((sizeof(input)/sizeof(int))/size_in)-2; n++)
            {
                
                for (int k = 0; k < ((sizeof(input)/sizeof(int))/size_in)-2; k++)
                {
                    layer_1_kern_sum = 0;
                    for (int i = 0; i<3; i++)
                    {
                        for (int j = 0; j<3; j++)
                        {
                            layer_1_kern_sum += learning_data[lear][i+n][j+k] * kernel_weight[i][j];
                        }
                    }
                    layer_1[n][k] = layer_1_kern_sum;
                    layer2 += layer_1_kern_sum * layer1_2_weight[n][k];
                    layer2_2 += layer_1_kern_sum * layer1_2_weight2[n][k];
                    }
            }
            
                error = 0;
            error2 = 0;
            double layer_1_delta[3][3];
            double layer_1_delta2[3][3];
            
            for (int i = 0; i < 3; i++)
            {
                for(int j = 0;j<3;j++ )
                {
                    error +=(goal_pred[lear] - layer2)* (goal_pred[lear] - layer2);
                    error2 +=(goal_pred2[lear] - layer2_2)* (goal_pred2[lear] - layer2_2);
                    
                    double layer2_delta = goal_pred[lear] - layer2;
                    double layer2_delta_2 = goal_pred2[lear] - layer2_2;
                    
                    layer_1_delta[i][j] = layer2_delta * layer1_2_weight[i][j];
                    layer_1_delta2[i][j] = layer2_delta_2 * layer1_2_weight2[i][j];
                   
                    
                    for (int n = 0; n<3; n++)
                    {
                         for (int k = 0; k < 3; k++)
                         {
                            kernel_weight[i][j] = kernel_weight[i][j] + ( lr_kernel * (learning_data[lear][i+n][j+k]*layer_1_delta[i][j]));
//                             kernel_weight[i][j] = kernel_weight[i][j] + ( lr_kernel * (learning_data[lear][i+n][j+k]*layer_1_delta2[i][j]));
                        }
                    }
                    layer1_2_weight[i][j] = layer1_2_weight[i][j] + (lr * (layer_1[i][j] * layer2_delta));
                    layer1_2_weight2[i][j] = layer1_2_weight2[i][j] + (lr * (layer_1[i][j] * layer2_delta_2));
                }
            }
//                    printf("Er1-%lf  Er2-%lf   lyr2_1 %lf lyr2_2 %lf \n\n", error,error2,layer2,layer2_2);
            double softmax1 =exp(layer2)/ (exp(layer2)+exp(layer2_2));
            double softmax2 = exp(layer2_2)/(exp(layer2)+exp(layer2_2));
            printf("Er1-%lf  Er2-%lf   lyr2_1 %lf lyr2_2 %lf \n\n", error,error2,softmax1,softmax2);
        }
        for (int i = 0; i < 3; i ++)
        {
            for (int j= 0; j < 3; j++)
            {
                printf( "%lf, " , kernel_weight[i][j]);
            }
            printf("\n");

        }
        
        printf("\n\n\n");
    }
    for (int i = 0; i < 3; i ++)
    {
        for (int j= 0; j < 3; j++)
        {
            printf( "%lf, " , layer1_2_weight[i][j]);
        }
        printf("\n");
        
      // test 

    }
    printf("\n");
    for (int i = 0; i < 3; i ++)
    {
        for (int j= 0; j < 3; j++)
        {
            printf( "%lf, " , layer1_2_weight2[i][j]);
        }
        printf("\n");
    }
    return 0;
}

