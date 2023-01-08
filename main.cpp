#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>  ///���ʹ��cuda���٣���Ҫȡ��ע��
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class PP_MattingV2
{
public:
	PP_MattingV2();
	Mat inference(Mat cv_image);
private:

	void preprocess(Mat srcimg);
	int inpWidth;
	int inpHeight;
	vector<float> input_image_;
	const float conf_threshold = 0.65;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "PP-MattingV2");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

PP_MattingV2::PP_MattingV2()
{
	string model_path = "weights/ppmattingv2_stdc1_human_480x640.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windowsд��
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///���ʹ��cuda���٣���Ҫȡ��ע��

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windowsд��
	////ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linuxд��

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void PP_MattingV2::preprocess(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);

	int row = dstimg.rows;
	int col = dstimg.cols;
	this->input_image_.resize(row * col * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

Mat PP_MattingV2::inference(Mat srcimg)
{
	this->preprocess(srcimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	// post process.																																					
	Value &mask_pred = ort_outputs.at(0);
	const int out_h = this->output_node_dims[0][2];
	const int out_w = this->output_node_dims[0][3];
	float *mask_ptr = mask_pred.GetTensorMutableData<float>();

	Mat segmentation_map;
	Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr);
	resize(mask_out, segmentation_map, Size(srcimg.cols, srcimg.rows));
	Mat dstimg = srcimg.clone();

	for (int h = 0; h < srcimg.rows; h++)
	{
		for (int w = 0; w < srcimg.cols; w++)
		{
			float pix = segmentation_map.ptr<float>(h)[w];
			if (pix > this->conf_threshold)
			{
				float b = (float)dstimg.at<Vec3b>(h, w)[0];
				dstimg.at<Vec3b>(h, w)[0] = uchar(b * 0.5 + 1);
				float g = (float)dstimg.at<Vec3b>(h, w)[1] + 255.0;
				dstimg.at<Vec3b>(h, w)[1] = uchar(g * 0.5 + 1);
				float r = (float)dstimg.at<Vec3b>(h, w)[2];
				dstimg.at<Vec3b>(h, w)[2] = uchar(r * 0.5 + 1);
			}
		}
	}
	return dstimg;
}

int main()
{
	const int use_video = 0;
	PP_MattingV2 mynet;
	if (use_video)
	{
		cv::VideoCapture video_capture(0);  ///���Ե�����ͷ
		//cv::VideoCapture video_capture(images/3.mp4);  ///��Ƶ�ļ�
		if (!video_capture.isOpened())
		{
			std::cout << "Can not open video " << endl;
			return -1;
		}

		cv::Mat frame;
		while (video_capture.read(frame))
		{
			Mat dstimg = mynet.inference(frame);
			string kWinName = "Deep learning ONNXRuntime with PP-MattingV2";
			namedWindow(kWinName, WINDOW_NORMAL);
			imshow(kWinName, dstimg);
			waitKey(1);
		}
		destroyAllWindows();
	}
	else
	{
		string imgpath = "images/3.jpg";
		Mat srcimg = imread(imgpath);
		Mat dstimg = mynet.inference(srcimg);

		namedWindow("srcimg", WINDOW_NORMAL);
		imshow("srcimg", srcimg);
		static const string kWinName = "Deep learning ONNXRuntime with PP-MattingV2";
		namedWindow(kWinName, WINDOW_NORMAL);
		imshow(kWinName, dstimg);
		waitKey(0);
		destroyAllWindows();
	}
}