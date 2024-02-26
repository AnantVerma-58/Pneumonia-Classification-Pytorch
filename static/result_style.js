// result_script.js

document.addEventListener('DOMContentLoaded', function () {
    // Function to add fade-in effect to result elements
    function fadeInElements() {
        const resultImage = document.querySelector('.result-image');
        const resultInfo = document.querySelectorAll('.uploaded-info, .prediction-info, .probability-info');

        // Add fade-in animation to the result image
        resultImage.style.animation = 'fadeIn 1s ease-in-out';

        // Add staggered fade-in animation to result info elements
        resultInfo.forEach((info, index) => {
            info.style.animation = `fadeIn 0.8s ease-in-out ${index * 0.2}s`;
        });
    }

    // Call the fadeInElements function when the DOM is loaded
    fadeInElements();
});
