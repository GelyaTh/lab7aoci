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
    public partial class Form1 : Form
    {
        private Image<Bgr, byte> baseImg; //глобальная переменная
        private Image<Bgr, byte> twistedImg; //глобальная переменная
        private Stabilizer stabilizer;


        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            var result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                baseImg = new Image<Bgr, byte>(fileName);

                imageBox1.Image = baseImg.Resize(640, 480, Inter.Linear);
            }

            result = openFileDialog.ShowDialog(); // открытие диалога выбора файла
            if (result == DialogResult.OK) // открытие выбранного файла
            {
                string fileName = openFileDialog.FileName;
                twistedImg = new Image<Bgr, byte>(fileName);

                imageBox2.Image = twistedImg.Resize(640, 480, Inter.Linear);
            }
            stabilizer = new Stabilizer(baseImg, twistedImg);
        }

        //ищет характерные точки и отображает на изображениях
        private void button2_Click(object sender, EventArgs e)
        {
            stabilizer.DrawOnImagesCharacteristicPoints();
            imageBox1.Image = stabilizer.outputImage1.Resize(640, 480, Inter.Linear);
            imageBox2.Image = stabilizer.outputImage2.Resize(640, 480, Inter.Linear);
        }

        //ищет гомографию и выравнивает изображение
        private void button3_Click(object sender, EventArgs e)
        {
            stabilizer.StabilizeImageWithLucasCanada();
            imageBox2.Image = stabilizer.outputImage2.Resize(640, 480, Inter.Linear);
        }

        //сравнение характерных точек
        private void button4_Click(object sender, EventArgs e)
        {
            stabilizer.StabilizeImageWithPointsComparison();
            imageBox3.Image = stabilizer.outputImage3;
            imageBox2.Image = stabilizer.outputImage2;
        }
    }
}
