package org.ml4j.nn.demos.inceptionv4;

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactoryOptimised;
import org.ml4j.nd4j.Nd4jRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.models.inceptionv4.InceptionV4Factory;
import org.ml4j.nn.models.inceptionv4.impl.DefaultInceptionV4Factory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactoryImpl;
import org.ml4j.nn.supervised.DefaultSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Conditional;
import org.springframework.context.annotation.Configuration;

@Configuration
public class InceptionV4Config {

	@Bean
	@Conditional(OSX_AArch64Condition.class)
	MatrixFactory matrixFactoryNd4j() {
		return new Nd4jRowMajorMatrixFactory();
	}


	@Bean
	@Conditional(NonOSX_AArch64Condition.class)
	MatrixFactory matrixFactoryJBlasOptimised() {
		return new JBlasRowMajorMatrixFactoryOptimised();
	}

	@Bean
	AxonsFactory axonsFactory(@Autowired MatrixFactory matrixFactory) {
		return new DefaultAxonsFactoryImpl(matrixFactory);
	}
	
	@Bean
	DirectedComponentFactory directedComponentFactory(@Autowired MatrixFactory matrixFactory) {
		return new DefaultDirectedComponentFactoryImpl(matrixFactory, axonsFactory(matrixFactory), activationFunctionFactory(),
				directedComponentsContext(matrixFactory));
	}
	
	@Bean
	DirectedComponentsContext directedComponentsContext(@Autowired MatrixFactory matrixFactory) {
		return new DirectedComponentsContextImpl(matrixFactory, false);
	}

	@Bean
	DefaultSessionFactory sessionFactory(@Autowired MatrixFactory matrixFactory) {
		return new DefaultSessionFactoryImpl(matrixFactory,
				directedComponentFactory(matrixFactory), null,  // No DirectedLayerFactory needed for this demo.
				supervisedFeedForwardNeuralNetworkFactory(matrixFactory), directedComponentsContext(matrixFactory));
	}

	@Bean
	DifferentiableActivationFunctionFactory activationFunctionFactory() {
		return new DefaultDifferentiableActivationFunctionFactory();
	}

	@Bean
	SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory(@Autowired MatrixFactory matrixFactory) {
		return new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory(matrixFactory));
	}
	
	@Bean
	InceptionV4Factory inceptionV4Factory(@Autowired MatrixFactory matrixFactory) throws IOException {
		return new DefaultInceptionV4Factory(sessionFactory(matrixFactory), matrixFactory,
				InceptionV4Demo.class.getClassLoader());
	}
}
