package com.xu.djl;


import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DjlTest {


    @Test
    public void testModel() throws MalformedModelException, IOException, TranslateException {
        //加载图片信息
        Image image = ImageFactory.getInstance().fromFile(Paths.get("build/img/2.jpg"));

        //加载模型
        Path modelPath = Paths.get("build/mlp");
        Model model = Model.newInstance("mlpxxx");
        model.setBlock(new Mlp(28*28, 10, new int[]{256, 128, 64}));
        model.load(modelPath);

        //3.预测
        //创建一个转换器，用来处理需要预测的图片，使得图片可以在模型中使用(将图片转换成NdArray)
        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(28, 28))
                .addTransform(new ToTensor())
                .build();
        //创建一个预测器
        Predictor<Image, Classifications> predictor = model.newPredictor(translator);

        Classifications predict = predictor.predict(image);
        System.out.println("预测结果 "+predict);

    }


    /**
     * 完全训练一个模型步骤：
     * 1.准备数据集
     * 2.构建神经网络
     * 3.构建模型(这个模型应用上面的神经网络)
     * 4.训练配置(如何训练、训练集、验证集、测试集)
     * 5.保存模型
     *
     * 训练好的模型使用:
     * 1.加载模型
     * 2.预测(给模型一个输入，让他判断这是什么)
     */
    @Test
    public void trainingComplete() throws IOException {
        //1.准备数据集(训练集、测试集)
        RandomAccessDataset trainDataset = getDataset(Dataset.Usage.TRAIN);
        RandomAccessDataset validDataset = getDataset(Dataset.Usage.TEST);
        //准备一个输出文件夹
        String outputDir = "build/mlp";

        //2.构建神经网络 block  Mlp(输入图片分辨率28x28, 输出10种预测结果, 隐藏层大小设置NdList)
        Mlp mlp = new Mlp(28*28, 10, new int[]{256, 128, 64});

        //3.创建模型，并且加载上面的神经网络
        try(Model model = Model.newInstance("mlpxxx")) {
            //加载神经网络
            model.setBlock(mlp);

            //4.训练神经网络 -->配置
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging(outputDir));

            //5.开始训练, 通过模型创建训练器
            try (Trainer trainer = model.newTrainer(config)){
                //训练时输出训练详细数据
                trainer.setMetrics(new Metrics());
                //初始化
                trainer.initialize(new Shape(1, 28*28));

                EasyTrain.fit(trainer, 10, trainDataset, validDataset);


                TrainingResult result = trainer.getTrainingResult();
                System.out.println("训练结果" +result);

                //5.保存模型
                model.save(Paths.get(outputDir), "mlpxxx");
                System.out.println("模型保存成功");

            } catch (TranslateException e) {
                throw new RuntimeException(e);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }






    }

    /**
     * 创建训练数据集
     * @param usage
     * @return
     */
    private RandomAccessDataset getDataset(Dataset.Usage usage) throws IOException {
        Mnist build = Mnist.builder()
                //用法训练/测试/验证
                .optUsage(usage)
                .setSampling(64, true)
                .build();
        //用于查看数据集加载的进度
        build.prepare(new ProgressBar());
        return build;
    }






    @Test
    void testDNarray() {
        //需要把处理过得初始数据转换成矩阵，然后通过隐藏层对这些矩阵进行处理，最后输出。

        //创建DNManager用于管理深度学习期间的临时数据。销毁后自动释放所有资源
        try (NDManager manager = NDManager.newBaseManager()) {

            //创建2x2矩阵,并初始化矩阵的数据
            NDArray ndArray = manager.create(new int[]{1, 2, 3, 4}, new Shape(2, 2));

            System.out.println(ndArray);

            ndArray.muli(2);
            System.out.println(ndArray);


        }

    }

}
