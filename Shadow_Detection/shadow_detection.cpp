// shadow_detection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#define PI 3.1415926


using namespace std;
using namespace cv;


Mat HSV_shadow(string file_name);
Mat C1C2C3_shadow(string file_name);

int main()
{
	string file_name = "Color.bmp";


	//HSV结果
	Mat HSV = HSV_shadow(file_name);
	imshow("HSV", HSV);
	waitKey();
	imwrite("HSV.jpg", HSV);


	//C1C2C3结果
	Mat C1C2C3 = C1C2C3_shadow(file_name);
	imshow("C1C2C3", C1C2C3);
	waitKey();
	imwrite("C1C2C3.jpg", C1C2C3);

	return 0;
}


Mat HSV_shadow(string file_name)
{
	//读入图像
	Mat image = imread(file_name, IMREAD_COLOR);


	int height = image.rows;
	int width = image.cols;


	Mat B(height, width, CV_8UC3);
	Mat G(height, width, CV_8UC3);
	Mat R(height, width, CV_8UC3);


	//将影像分为三个波段
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			B.at<uchar>(i, j) = image.at<Vec3b>(i, j)[0];
			G.at<uchar>(i, j) = image.at<Vec3b>(i, j)[1];
			R.at<uchar>(i, j) = image.at<Vec3b>(i, j)[2];
		}
	}

	//以浮点数形式存储
	//Store pixel as double
	double* b = new double[height * width];
	double* g = new double[height * width];
	double* r = new double[height * width];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			b[i * width + j] = (double)(B.at<uchar>(i, j) / 255.0);
			g[i * width + j] = (double)(G.at<uchar>(i, j) / 255.0);
			r[i * width + j] = (double)(R.at<uchar>(i, j) / 255.0);
		}
	}


	//初始化参数
	double* H = new double[height * width];
	double* S = new double[height * width];
	double* V = new double[height * width];
	double* theta = new double[height * width];
	double* M = new double[height * width];


	//计算V分量
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			V[i * width + j] = (double)((b[i * width + j] + g[i * width + j] + r[i * width + j]) / 3.0 * 180.0);
		}
	}


	//计算S分量
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (b[i * width + j] + g[i * width + j] + r[i * width + j] != 0)
			{
				S[i * width + j] = (double)(((double)(1.0) - (double)(3.0 * min(min(b[i * width + j], g[i * width + j]), r[i * width + j]) /
					(b[i * width + j] + g[i * width + j] + r[i * width + j]))) * 180.0);
			}
		}
	}


	//计算theta
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (sqrt(pow((r[i * width + j] - g[i * width + j]), 2) + (r[i * width + j] - b[i * width + j]) * (g[i * width + j] - b[i * width + j])) != 0)
			{
				theta[i * width + j] = (double)(acos(0.5 * ((r[i * width + j] - g[i * width + j]) + (r[i * width + j] + b[i * width + j])) /
					sqrt(pow((r[i * width + j] - g[i * width + j]), 2) + (r[i * width + j] - b[i * width + j]) * (g[i * width + j] - b[i * width + j]))) * 180 / PI);
			}
		}
	}


	//计算H分量
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (b[i * width + j] <= g[i * width + j])
			{
				H[i * width + j] = theta[i * width + j];
			}
			else
			{
				H[i * width + j] = (double)(360.0 - theta[i * width + j]);
			}
		}
	}


	//计算参数矩阵M
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			M[i * width + j] = (double)((S[i * width + j] - V[i * width + j]) / (H[i * width + j] + S[i * width + j] + V[i * width + j]));
		}
	}


	//参数矩阵后处理
	Mat M_final = Mat::zeros(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//排除未知区
			if (image.at<Vec3b>(i, j)[0] == 34 && image.at<Vec3b>(i, j)[1] == 34 && image.at<Vec3b>(i, j)[2] == 50)
			{
				M_final.at<uchar>(i, j) = 0;
			}
			//排除白色区
			else if (image.at<Vec3b>(i, j)[0] == 255 && image.at<Vec3b>(i, j)[1] == 255 && image.at<Vec3b>(i, j)[2] == 255)
			{
				M_final.at<uchar>(i, j) = 0;
			}
			//设定阈值二值化
			else
			{
				if (M[i * width + j] > 0.35)
				{
					M_final.at<uchar>(i, j) = 255;
				}
				else
				{
					M_final.at<uchar>(i, j) = 0;
				}
			}
		}
	}


	delete[] b;
	delete[] g;
	delete[] r;

	delete[] H;
	delete[] S;
	delete[] V;
	delete[] theta;
	delete[] M;


	return M_final;
}


Mat C1C2C3_shadow(string file_name)
{
	//读入图像
	Mat image = imread(file_name, IMREAD_COLOR);


	int height = image.rows;
	int width = image.cols;


	Mat B(height, width, CV_8UC3);
	Mat G(height, width, CV_8UC3);
	Mat R(height, width, CV_8UC3);


	//将影像分为三个波段
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			B.at<uchar>(i, j) = image.at<Vec3b>(i, j)[0];
			G.at<uchar>(i, j) = image.at<Vec3b>(i, j)[1];
			R.at<uchar>(i, j) = image.at<Vec3b>(i, j)[2];
		}
	}


	//以浮点数形式存储
	double* b = new double[height * width];
	double* g = new double[height * width];
	double* r = new double[height * width];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			b[i * width + j] = (double)(B.at<uchar>(i, j) / 255.0);
			g[i * width + j] = (double)(G.at<uchar>(i, j) / 255.0);
			r[i * width + j] = (double)(R.at<uchar>(i, j) / 255.0);
		}
	}


	//初始化参数
	double* C3 = new double[height * width];


	//计算C3分量
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			C3[i * width + j] = atan(b[i * width + j] / max(r[i * width + j], g[i * width + j]));
		}
	}


	//C3分量后处理
	Mat C3_final = Mat::zeros(height, width, CV_8UC1);


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//排除未知区
			if (image.at<Vec3b>(i, j)[0] == 34 && image.at<Vec3b>(i, j)[1] == 34 && image.at<Vec3b>(i, j)[2] == 50)
			{
				C3_final.at<uchar>(i, j) = 0;
			}
			//双阈值法二值化
			else
			{
				if (C3[i * width + j] > 0.1 && b[i * width + j] < 0.1)
				{
					C3_final.at<uchar>(i, j) = 255;
				}
				if (C3[i * width + j] <= 0.1 || b[i * width + j] >= 0.1)
				{
					C3_final.at<uchar>(i, j) = 0;
				}
			}
		}
	}


	delete[] b;
	delete[] g;
	delete[] r;

	delete[] C3;


	return C3_final;
}


// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件