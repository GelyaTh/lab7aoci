using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using Emgu.CV.Features2D;

namespace Lab7_AOCI
{
    class Stabilizer
    {
        private Image<Bgr, byte> baseImg; //глобальная переменная
        private Image<Bgr, byte> twistedImg; //глобальная переменная
        public Image<Bgr, byte> outputImage1;
        public Image<Bgr, byte> outputImage2;
        public Image<Bgr, byte> outputImage3;

        GFTTDetector detector = new GFTTDetector(40, 0.01, 5, 3, true);

        public Stabilizer(Image<Bgr, byte> baseImg, Image<Bgr, byte> twistedImg)
        {
            this.baseImg = baseImg;
            this.twistedImg = twistedImg;
            this.outputImage1 = this.baseImg.Clone();
            this.outputImage2 = this.twistedImg.Clone();
        }

        private void FindCharacteristicPoints(bool drawOnImage, bool warpImage)
        {

            MKeyPoint[] GFP1 = detector.Detect(baseImg.Convert<Gray, byte>().Mat);

            //создание массива характерных точек исходного изображения (только позиции)
            PointF[] srcPoints = new PointF[GFP1.Length];
            for (int i = 0; i < GFP1.Length; i++)
                srcPoints[i] = GFP1[i].Point;

            PointF[] destPoints; //массив для хранения позиций точек на изменённом изображении

            byte[] status; //статус точек (найдены/не найдены)
            float[] trackErrors; //ошибки

            //вычисление позиций характерных точек на новом изображении методом Лукаса-Канаде
            CvInvoke.CalcOpticalFlowPyrLK(
                    baseImg.Convert<Gray, byte>().Mat, //исходное изображение
                    twistedImg.Convert<Gray, byte>().Mat,//изменённое изображение
                    srcPoints, //массив характерных точек исходного изображения
                    new Size(20, 20), //размер окна поиска
                    5, //уровни пирамиды
                    new MCvTermCriteria(20, 1), //условие остановки вычисления оптического потока
                    out destPoints, //позиции характерных точек на новом изображении
                    out status, //содержит 1 в элементах, для которых поток был найден
                    out trackErrors //содержит ошибки
                );

            if (warpImage)
            {
                var destImage = new Image<Bgr, byte>(baseImg.Size);

                //вычисление матрицы гомографии
                Mat homographyMatrix = CvInvoke.FindHomography(destPoints, srcPoints, RobustEstimationAlgorithm.LMEDS);
                CvInvoke.WarpPerspective(twistedImg, destImage, homographyMatrix, destImage.Size);

                foreach (PointF p in destPoints)
                {
                    CvInvoke.Circle(destImage, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 2);
                }

                outputImage2 = destImage.Resize(640, 480, Inter.Linear);
            }

            if (drawOnImage)
            {
                var output1 = baseImg.Clone();

                foreach (MKeyPoint p in GFP1)
                {
                    CvInvoke.Circle(output1, Point.Round(p.Point), 3, new Bgr(Color.Blue).MCvScalar, 2);
                }

                var output2 = twistedImg.Clone();

                foreach (PointF p in destPoints)
                {
                    CvInvoke.Circle(output2, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 2);
                }

                outputImage1 = output1.Resize(640, 480, Inter.Linear);
                outputImage2 = output2.Resize(640, 480, Inter.Linear);
            }
        }

        public void StabilizeImageWithPointsComparison()
        {
            var baseImgGray = baseImg.Convert<Gray, byte>();
            var twistedImgGray = twistedImg.Convert<Gray, byte>();

            //генератор описания ключевых точек
            Brisk descriptor = new Brisk();

            //поскольку в данном случае необходимо посчитать обратное преобразование
            //базой будет являться изменённое изображение
            VectorOfKeyPoint GFP1 = new VectorOfKeyPoint();
            UMat baseDesc = new UMat();
            UMat bimg = twistedImgGray.Mat.GetUMat(AccessType.Read);

            VectorOfKeyPoint GFP2 = new VectorOfKeyPoint();
            UMat twistedDesc = new UMat();
            UMat timg = baseImgGray.Mat.GetUMat(AccessType.Read);

            //получение необработанной информации о характерных точках изображений
            detector.DetectRaw(bimg, GFP1);
            //генерация описания характерных точек изображений
            descriptor.Compute(bimg, GFP1, baseDesc);

            detector.DetectRaw(timg, GFP2);
            descriptor.Compute(timg, GFP2, twistedDesc);


            //класс позволяющий сравнивать описания наборов ключевых точек
            BFMatcher matcher = new BFMatcher(DistanceType.L2);

            //массив для хранения совпадений характерных точек
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

            //добавление описания базовых точек
            matcher.Add(baseDesc);
            //сравнение с описанием изменённых
            matcher.KnnMatch(twistedDesc, matches, 2, null);
            //3й параметр - количество ближайших соседей среди которых осуществляется поиск совпадений //4й параметр - маска, в данном случае не нужна


            //маска для определения отбрасываемых значений (аномальных и не уникальных)
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            //определение уникальных совпадений 
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);


            Mat homography;
            //получение матрицы гомографии 
            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(GFP1, GFP2, matches, mask, 2);

            var res = new Image<Bgr, byte>(baseImg.Size);
            var destImage = new Image<Bgr, byte>(baseImg.Size);
            CvInvoke.WarpPerspective(twistedImg, destImage, homography, destImage.Size);

            Features2DToolbox.DrawMatches(twistedImg, GFP1, baseImg, GFP2, matches, res, new MCvScalar(255, 0, 0), new MCvScalar(255, 0, 0), mask);

            outputImage3 = res.Resize(1286, 480, Inter.Linear);
            outputImage2 = destImage.Resize(640, 480, Inter.Linear);
        }

        public void StabilizeImageWithLucasCanada()
        {
            FindCharacteristicPoints(false, true);
        }

        public void DrawOnImagesCharacteristicPoints()
        {
            FindCharacteristicPoints(true, false);
        }
    }
}
