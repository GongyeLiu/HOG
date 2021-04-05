#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#define PI 3.1415926535
using namespace std;
using namespace cv;


// Gamma校正
void gammaTrans(Mat src, Mat &dst, double gamma)
{
	src.convertTo(dst, CV_32FC1, 1.0 / 255);
	pow(dst, gamma, dst);
	dst.convertTo(dst, CV_8UC1, 255.0);
	return;
}

// 计算图像的梯度图
void calcGrad(Mat src, Mat &dst)
{
	dst = Mat::zeros(src.cols, src.rows, CV_8UC1);
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			double dx, dy;
			int theta;
			dx = src.at<uchar>(i, j + 1) - src.at<uchar>(i, j - 1);	// 计算水平方向和垂直方向梯度
			dy = src.at<uchar>(i + 1, j) - src.at<uchar>(i - 1, j);
			theta = atan2(dy, dx) * 180.0 / PI;		//计算梯度方向
			if (theta < 0)
				theta += 180;
			if (theta == 180)
				theta = 0;
			dst.at<uchar>(i, j) = theta;
		}
	}
	return;
}


// 统计cell中的梯度直方图
void calcHist(Mat gradImg, vector<int> &hist)
{
	hist.clear();
	// 初始化9个bins为0
	for (int i = 0; i < 9; i++)
	{
		hist.push_back(0);
	}

	// 统计直方图
	for (int i = 0; i < gradImg.rows; i++)
	{
		for (int j = 0; j < gradImg.cols; j++)
		{
			int theta = gradImg.at<uchar>(i, j);
			hist[theta / 20]++;
		}
	}
}


// 绘制一个cell的HOG特征
void drawCellHog(Mat &dst, vector<int> feature)
{
	dst = Mat::zeros(8, 8, CV_8UC1);	// 初始化特征图
	for (int i = 0; i < feature.size(); i++)
	{
		int l = round(feature[i] / 64.0 * 4);	// l为当前方向的长度，即将0-64的bins映射到0-4的长度
		double theta = i * 20 + 10;		// 当前梯度方向的角度
		theta = theta * PI / 180.0;
		// 绘制长度为l，角度为theta的直线
		for (int j = 0; j < l; j++)
		{
			int x = round(j * cos(theta));
			int y = round(j * sin(theta));
			dst.at<uchar>(3 + x, 3 + y) = 255;
			dst.at<uchar>(3 - x, 3 - y) = 255;
			/*dst.at<uchar>(3 + x, 4 + y) = 255;
			dst.at<uchar>(3 - x, 4 - y) = 255;
			dst.at<uchar>(4 + x, 3 + y) = 255;
			dst.at<uchar>(4 - x, 3 - y) = 255;
			dst.at<uchar>(4 + x, 4 + y) = 255;
			dst.at<uchar>(4 - x, 4 - y) = 255;*/
		}
	}
}


// 获取该block的HOG特征
void getBlockHog(Mat gradImg, vector<int> &result,  Mat &hogImg, int cellSize)
{
	hogImg = Mat::zeros(gradImg.cols, gradImg.rows, CV_8UC1);
	// 依次计算每个cell的HOG特征
	for (int i = 0; i < gradImg.cols; i += cellSize)
	{
		for (int j = 0; j < gradImg.rows; j += cellSize)
		{
			vector<int> feature;
			Mat roiImg;
			Rect ROI = Rect(i, j, cellSize, cellSize);
			calcHist(gradImg(ROI), feature);	// 计算当前cell的梯度直方图
			for (auto t : feature)		// 添加到结果
			{
				result.push_back(t);
			}
			drawCellHog(roiImg, feature);	// 绘制当前cell的HOG特征
			roiImg.copyTo(hogImg(ROI));
		}
	}
}


// 计算HOG特征并绘制特征图
void hog(Mat image, vector<int> &result, Mat &hogImg, int cellSize = 8, int blockSize = 2, int step = 8)
{
	Mat gradImg;
	hogImg = Mat::zeros(image.cols, image.rows, CV_8UC1);	// 初始化HOG特征图
	result.clear();		// 初始化特征向量
	gammaTrans(image, image, 1.2);		// Gamma校正
	calcGrad(image, gradImg);			// 计算整幅图像的梯度图
	 
	int block_xy = cellSize * blockSize; // block的像素大小
	// 依次计算每个block的hog特征向量并绘制特征图
	for (int i = 0; i + block_xy <= image.cols; i += step)
	{
		for (int j = 0; j + block_xy <= image.rows; j += step)
		{ 
			Mat roiImg;
			Rect ROI = Rect(i, j, block_xy, block_xy);
			getBlockHog(gradImg(ROI), result, roiImg, cellSize);
			roiImg.copyTo(hogImg(ROI));
		}
	}
}
int main()
{
	Mat image, hogImg;
	vector<int> result;
	image = imread("2.jpeg", IMREAD_GRAYSCALE);

	namedWindow("origin");
	namedWindow("HOG");
	imshow("origin", image);
	
	hog(image, result, hogImg);
	imshow("HOG", hogImg);
	cout << result.size() << endl;
	waitKey(0);
	destroyAllWindows();
	imwrite("2_hog.jpg", image);
	system("Pause");
	return 0;
}