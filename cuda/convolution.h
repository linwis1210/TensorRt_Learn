#pragma once

#include <cstddef>
#include <cstdlib>

#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <cudnn.h>
#include <cudnn/reference.h>
#include <cudnn/context.h>
#include <cudnn/utils_profile.h>
#include <cudnn/debug.h>

class ConvolutionTester {
public:
	ConvolutionTester() :
		iterations_(1),
		errorLimit_(1.0e-5),
		multithreading_(false),
		bias_(true),
		batchSize_(1),
		scaleFactor_(1),
		inputChannels_(1),
		outputChannels_(1)
	{
		inputSize(4, 4);
		kernelSize(3, 3);
		inputPadding(0, 0, 0, 0);
		outputSubsampling(1, 1);

		assert(cudnnCreate(&(this->cudnnHandle), 1) == CUDNN_STATUS_SUCCESS);
	}

	ConvolutionTester(const ConvolutionTester&) = delete;

	inline ConvolutionTester(ConvolutionTester&& tester) :
		iterations_(tester.iterations_),
		errorLimit_(tester.errorLimit_),
		multithreading_(tester.multithreading_),
		bias_(tester.bias_),
		batchSize_(tester.batchSize_),
		scaleFactor_(tester.scaleFactor_),
		inputChannels_(tester.inputChannels_),
		outputChannels_(tester.outputChannels_),
		inputSize_(tester.inputSize_),
		inputPadding_(tester.inputPadding_),
		kernelSize_(tester.kernelSize_),
		outputSubsampling_(tester.outputSubsampling_),
		cudnnHandle(tester.cudnnHandle)
	{
		tester.cudnnHandle = nullptr;
	}

	ConvolutionTester& operator=(const ConvolutionTester&) = delete;

	~ConvolutionTester() {
		if (this->cudnnHandle != nullptr) {
			cudnnDestroy(this->cudnnHandle);
			this->cudnnHandle = nullptr;
		}
	}

	inline ConvolutionTester& iterations(size_t iterations) {
		this->iterations_ = iterations;
		return *this;
	}

	inline size_t iterations() const {
		return this->iterations_;
	}

	inline ConvolutionTester& errorLimit(float errorLimit) {
		this->errorLimit_ = errorLimit;
		return *this;
	}

	inline float errorLimit() const {
		return this->errorLimit_;
	}

	inline ConvolutionTester& multithreading(bool multithreading) {
        this->multithreading_ = multithreading;
        if (multithreading && this->cudnnHandle != nullptr) {
			cudnnDestroy(this->cudnnHandle);
            assert(cudnnCreate(&(this->cudnnHandle), 0) == CUDNN_STATUS_SUCCESS);
        } else if (multithreading && this->cudnnHandle == nullptr) {
            assert(cudnnCreate(&(this->cudnnHandle), 0) == CUDNN_STATUS_SUCCESS);
        } else if (!multithreading && this->cudnnHandle != nullptr) {
            cudnnDestroy(this->cudnnHandle);
            assert(cudnnCreate(&(this->cudnnHandle), 1) == CUDNN_STATUS_SUCCESS);
        }
        return *this;
	}

	inline bool multithreading() const {
		return this->multithreading_;
	}

	inline ConvolutionTester& bias(bool bias) {
		this->bias_ = bias;
		return *this;
	}

	inline bool bias() const {
		return this->bias_;
	}

	inline ConvolutionTester& batchSize(size_t batchSize) {
		this->batchSize_ = batchSize;
		return *this;
	}

	inline size_t batchSize() const {
		return this->batchSize_;
	}

	inline ConvolutionTester& scaleFactor(size_t scaleFactor) {
		this->scaleFactor_ = scaleFactor;
		return *this;
	}

	inline size_t scaleFactor() const {
		return this->scaleFactor_;
	}

	inline ConvolutionTester& inputChannels(size_t inputChannels) {
		this->inputChannels_ = inputChannels;
		return *this;
	}

	inline size_t inputChannels() const {
		return this->inputChannels_;
	}

	inline ConvolutionTester& outputChannels(size_t outputChannels) {
		this->outputChannels_ = outputChannels;
		return *this;
	}

	inline size_t outputChannels() const {
		return this->outputChannels_;
	}

	inline ConvolutionTester& inputSize(size_t height, size_t width) {
		this->inputSize_.height = height;
		this->inputSize_.width = width;
		return *this;
	}

	inline struct cudnn_size inputSize() const {
		return this->inputSize_;
	}

	inline size_t inputHeight() const {
		return this->inputSize_.height;
	}

	inline size_t inputWidth() const {
		return this->inputSize_.width;
	}

	inline ConvolutionTester& kernelSize(size_t height, size_t width) {
		this->kernelSize_.height = height;
		this->kernelSize_.width = width;
		return *this;
	}

	inline struct cudnn_size kernelSize() const {
		return this->kernelSize_;
	}

	inline size_t kernelHeight() const {
		return this->kernelSize_.height;
	}

	inline size_t kernelWidth() const {
		return this->kernelSize_.width;
	}

	inline struct cudnn_size outputSize() const {
		struct cudnn_size outputSize;
		outputSize.height = this->outputHeight();
		outputSize.width = this->outputWidth();
		return outputSize;
	}

	inline size_t outputHeight() const {
		return (this->inputPadding_.top + this->inputSize_.height + this->inputPadding_.bottom - this->kernelSize_.height) / this->outputSubsampling_.height + 1;
	}

	inline size_t outputWidth() const {
		return (this->inputPadding_.left + this->inputSize_.width + this->inputPadding_.right - this->kernelSize_.width) / this->outputSubsampling_.width + 1;
	}

	inline ConvolutionTester& outputSubsampling(size_t height, size_t width) {
		this->outputSubsampling_.height = height;
		this->outputSubsampling_.width = width;
		return *this;
	}

	inline struct cudnn_size outputSubsampling() const {
		return this->outputSubsampling_;
	}

	inline ConvolutionTester& inputPadding(size_t top, size_t right, size_t bottom, size_t left) {
		this->inputPadding_.top = top;
		this->inputPadding_.right = right;
		this->inputPadding_.bottom = bottom;
		this->inputPadding_.left = left;
		return *this;
	}

	inline struct cudnn_padding inputPadding() const {
		return this->inputPadding_;
	}

	void testFwd(size_t memoryLimitInBytes = 0x100000000, cudnnConvolutionFwdAlgo_t algo_i = CUDNN_CONVOLUTION_FWD_ALGO_GEMM, 
		cudnnTensorFormat_t input_Format = CUDNN_TENSOR_NCHW, 
		cudnnTensorFormat_t kernel_Format = CUDNN_TENSOR_NCHW, 
		cudnnTensorFormat_t output_Format = CUDNN_TENSOR_NCHW, 
		cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY) const {
        // Activation Tensor Initialize
		const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
		#if 1
		auto rng = std::bind(std::uniform_real_distribution<float>(-0.1f, 1.0f), std::mt19937(seed));
		#else
		auto rng = std::bind(std::uniform_real_distribution<float>(1.0f, 1.0f), std::mt19937(seed));
		#endif
		
		size_t input_size = batchSize() * inputChannels() * inputHeight() * inputWidth();
		size_t kernel_size = outputChannels() * inputChannels() * kernelHeight() * kernelWidth();
		size_t output_size = batchSize() * outputChannels() * outputHeight() * outputWidth();
		CUDNN_TEST("Input size(B=%zu, C=%zu, Hi=%zu, Wi=%zu)\n", batchSize(), inputChannels(), inputHeight(), inputWidth());
		CUDNN_TEST("Kernel size(K=%zu, C=%zu, Hf=%zu, Wf=%zu)\n", outputChannels(), inputChannels(), kernelHeight(), kernelWidth());
		CUDNN_TEST("Output size(B=%zu, K=%zu, Ho=%zu, Wo=%zu)\n", batchSize(), outputChannels(), outputHeight(), outputWidth());
		
		std::vector<float> input(input_size);
		std::vector<float> kernel(kernel_size);
		std::vector<float> output(output_size);
		std::vector<float> referenceOutput(output_size);
		void* tempSpace = NULL;
		size_t scratchSize = 0;
		cudnnConvolutionFwdAlgo_t algo = algo_i;

        // Status
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;

        // Set Descriptor
        cudnnTensorDescriptor_t      input_desc;
        cudnnTensorDescriptor_t      output_desc;
        cudnnFilterDescriptor_t      filter_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
		
		if(input_Format == CUDNN_TENSOR_NCHW) {
			cudnnSetTensor4dDescriptor(input_desc,
									input_Format,
									CUDNN_DATA_FLOAT,
									batchSize(),
									inputChannels(),
									inputHeight(),
									inputWidth());
			cudnnSetTensor4dDescriptor(output_desc,
									output_Format,
									CUDNN_DATA_FLOAT,
									batchSize(),
									outputChannels(),
									outputHeight(),
									outputWidth());
	        cudnnSetFilter4dDescriptor(filter_desc,
                                   CUDNN_DATA_FLOAT,
                                   kernel_Format,
                                   outputChannels(),
                                   inputChannels(),
                                   kernelHeight(),
                                   kernelWidth());          
        } else {
			CUDNN_TEST("Input Format is not supported now\n");
			cudnnDestroyTensorDescriptor(input_desc);
			cudnnDestroyTensorDescriptor(output_desc);
			cudnnDestroyFilterDescriptor(filter_desc);
			cudnnDestroyConvolutionDescriptor(conv_desc);
			return ;	
		}
		CUDNN_TEST("stride=%zu, %zu, Padding=%zu, Padding=%zu\n", 
			outputSubsampling().width, outputSubsampling().height, 
			inputPadding().top, inputPadding().left);
        cudnnSetConvolution2dDescriptor(conv_desc,
                                        inputPadding().top,
                                        inputPadding().left,
                                        outputSubsampling().width,
                                        outputSubsampling().height,
                                        1,
                                        1,
                                        CUDNN_CONVOLUTION,
                                        CUDNN_DATA_FLOAT);

		if(memoryLimitInBytes != 0) {
			// Get Forward algorithm
			CUDNN_TEST("memoryLimitInBytes=%zu\n", memoryLimitInBytes);
			status = cudnnGetConvolutionForwardAlgorithm((this->cudnnHandle),
										input_desc,
										filter_desc,
										conv_desc,
										output_desc,
										CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
										memoryLimitInBytes,
										&algo);
			ASSERT_EQ(CUDNN_STATUS_SUCCESS, status);
		}

        // Get Forward Workspace Size
        status = cudnnGetConvolutionForwardWorkspaceSize((this->cudnnHandle),
                                                         input_desc,
                                                         filter_desc,
                                                         conv_desc,
                                                         output_desc,
                                                         algo,
                                                         &scratchSize);
		CUDNN_TEST("scratchSize = %zu after cudnnGetConvolutionForwardWorkspaceSize, scaleFactor()=%zu!\n", scratchSize, scaleFactor());
		scratchSize *= scaleFactor();
		if(scratchSize != 0) {
			tempSpace = malloc(scratchSize);
			if(tempSpace == NULL) {
				CUDNN_TEST("malloc(scratchSize = %zu) error !\n", scratchSize);
			} else {
				CUDNN_TEST("malloc(scratchSize = %zu) success!\n", scratchSize);
			}
		}
		
		std::vector<float> maxErrors;
		#if CUDNN_PROFILES_ON
		struct cudnn_profile computation_profile[iterations()];
		#endif /*CUDNN_PROFILES_ON*/

		for (size_t iteration = 0; iteration < iterations(); iteration++) {
			CUDNN_TEST("--------------------Ite=%zu----------------\n", iteration);
			std::generate(input.begin(), input.end(), std::ref(rng));
			std::generate(kernel.begin(), kernel.end(), std::ref(rng));
			
			if(scratchSize != 0) {
				cudnn_fill_fp32((float *)tempSpace, scratchSize/sizeof(float), 0.0f);
			}	

			cudnn_convolution_output_fp32__reference(
				batchSize(), inputChannels(), outputChannels(),
				inputSize(), inputPadding(), kernelSize(), outputSubsampling(),
				input.data(), input_Format,
				kernel.data(), kernel_Format,
				nullptr,
				referenceOutput.data(), output_Format,
				this->cudnnHandle);

			float alpha = 1.0;
			float beta = 0.0;
			status = cudnnConvolutionForward((this->cudnnHandle),
										&alpha,
										input_desc, input.data(),
										filter_desc, kernel.data(),
										conv_desc, algo,
										tempSpace, scratchSize,
										&beta, 
										output_desc, output.data());
			ASSERT_EQ(CUDNN_STATUS_SUCCESS, status);
			#if CUDNN_PROFILES_ON
			computation_profile[iteration] = *((this->cudnnHandle)->profile);
			#endif /*CUDNN_PROFILES_ON*/
			check(output.data(), referenceOutput.data(), output_size);
			const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);
			maxErrors.push_back(maxError);
		}
		#if CUDNN_PROFILES_ON
		struct cudnn_profile  relu_profile = median_profile(computation_profile, iterations());
		printf_cudnn_profile(&relu_profile);
		#endif /*CUDNN_PROFILES_ON*/

		EXPECT_LT(median(maxErrors), errorLimit());

	    cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
		if(tempSpace != NULL) {
			free(tempSpace);
		}
	}


protected:
	cudnnHandle_t cudnnHandle;

private:
	inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

	inline static float median(std::vector<float>& array) {
		std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
		return array[array.size() / 2];
	}

    inline static float absError(float reference, float actual) {
        return std::abs(reference - actual);
    }

	size_t scaleFactor_;
	size_t iterations_;
	float errorLimit_;
	bool multithreading_;
	bool bias_;

	size_t batchSize_;
	size_t inputChannels_;
	size_t outputChannels_;
	struct cudnn_size inputSize_;
	struct cudnn_padding inputPadding_;
	struct cudnn_size kernelSize_;
	struct cudnn_size outputSubsampling_;
};
