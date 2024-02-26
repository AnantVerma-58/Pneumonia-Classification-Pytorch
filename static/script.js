// static/script.js

document.addEventListener('DOMContentLoaded', function () {
    // Get the upload form and sample form
    const uploadForm = document.querySelector('.upload-section form');
    const sampleForm = document.getElementById('sampleForm');

    // Attach submit event listeners to the forms
    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        // Handle the upload form submission
        submitUploadForm(uploadForm);
    });

    document.getElementById('sampleButton').addEventListener('click', function () {
        // Handle the sample form submission
        submitSampleForm();
    });
});

function submitUploadForm(form) {
    const formData = new FormData(form);
    fetch('/uploadfile/', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (response.ok) {
            return response.text();  // Use response.text() for HTML content
        } else {
            throw new Error(`Server responded with status ${response.status}`);
        }
    })
    .then(htmlContent => {
        // Update the current page with the HTML content
        document.documentElement.innerHTML = htmlContent;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


function submitSampleForm() {
    const selectedSample = document.getElementById('sample_image').value;
    // Set the value of the sampleNumber dropdown
    document.getElementById('sample_image').value = selectedSample;
    // Submit the form
    document.getElementById('sampleForm').submit();
}
