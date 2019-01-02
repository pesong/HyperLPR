#include "../include/SegmentationFreeRecognizer.h"
#include "../include/Pipeline.h"
#include "../include/PlateInfo.h"
#include "../include/CvxText.hpp"

#include <iostream>
#include <boost/algorithm/string.hpp>


int main(){
    pr::PipelinePR prc("/home/gs/code/HyperLPR/Prj-Linux/lpr/model/cascade.xml",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/HorizonalFinemapping.prototxt",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/HorizonalFinemapping.caffemodel",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/Segmentation.prototxt",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/Segmentation.caffemodel",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/CharacterRecognization.prototxt",
                      "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/CharacterRecognization.caffemodel",
                       "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/SegmenationFree-Inception.prototxt",
                       "/home/gs/code/HyperLPR/Prj-Linux/lpr/model/SegmenationFree-Inception.caffemodel"
                    );

    cv::VideoCapture cap;
//    cap.open(1);
    cap.open("/home/gs/code/HyperLPR/Prj-Linux/detect.mp4");

    if(!cap.isOpened()){
        exit(-1);
    }

    //视频写入对象
    cv::VideoWriter videoWriter;
    //写入视频文件名
    std::string outFlie = "/home/gs/code/HyperLPR/Prj-Linux/detect_out.mp4";
    //获得帧的宽高
    int w = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::Size S(w, h);
    //获得帧率
    double r = cap.get(CV_CAP_PROP_FPS);
    //打开视频文件，准备写入
    videoWriter.open(outFlie, CV_FOURCC('D', 'I', 'V', 'X'), 24, S, true);

    bool stop = false;

    cv::Mat frame;
    CvxText text("/home/gs/code/HyperLPR/Prj-Linux/lpr/src/SimHei.ttf");

    cv::Scalar size1{ 100, 0.5, 0.1, 0 }, size2{ 100, 0, 0.1, 0 }, size3{ 50, 0, 1, 0 }, size4{20, 0, 0.1, 0};
    text.setFont(nullptr, &size4, nullptr, 0);



    while(!stop){

//        cv::Mat frame = cv::imread("/home/gs/code/HyperLPR/Prj-Linux/lpr/1.jpg");
        cap.read(frame);
        std::vector<pr::PlateInfo> res = prc.RunPiplineAsImage(frame, pr::SEGMENTATION_FREE_METHOD);

        for(int i = 0; i < res.size(); i++){
            if(res[i].confidence>0.95) {
                std::cout << res[i].getPlateName() << " " << res[i].confidence << std::endl;

                std::string label_result = res[i].getPlateName();
                char * text_char = new char [label_result.length()+1];
                char * text_char_new = new char [label_result.length()+2];
                strcpy (text_char, label_result.c_str());

                text_char_new[0] = text_char[0];
                text_char_new[1] = text_char[1];
                text_char_new[2] = text_char[2];
                text_char_new[3] = text_char[2];

                for(int i=3; i< label_result.length()+1; i++){
                    text_char_new[i+1] = text_char[i];
                }

                //get detection result
                cv::Rect region = res[i].getPlateRect();

                // draw bbox
                cv::rectangle(frame,cv::Point(region.x,region.y),
                              cv::Point((region.x+region.width * 0.9),(region.y+region.height * 0.9)),cv::Scalar(255,0,0),1);
//
//                cv::putText(frame,"沪", cv::Point(region.x, region.y),
//                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 0.4, CV_AA);

                //draw text
                text.putText(frame, text_char_new, cv::Point(region.x, region.y), cv::Scalar(255, 255, 255));

            }
        }

        cv::imshow("Video",frame);
        videoWriter.write(frame);
        if(cv::waitKey(5)>0){
            stop = true;
            break;
        }
    }

    //释放对象
    cap.release();
    videoWriter.release();
    cvDestroyWindow("Video");

}
