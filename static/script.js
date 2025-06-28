document.addEventListener('DOMContentLoaded', () => {
    const reviewText = document.getElementById('reviewText');
    const predictButton = document.getElementById('predictButton');
    const sentimentSpan = document.getElementById('sentiment');
    const confidenceSpan = document.getElementById('confidence');
    const errorMessageDiv = document.getElementById('errorMessage');

    predictButton.addEventListener('click', async () => {
        const text = reviewText.value.trim(); // Get text from textarea and trim whitespace

        // Clear previous results and errors
        sentimentSpan.textContent = 'N/A';
        confidenceSpan.textContent = 'N/A';
        errorMessageDiv.style.display = 'none';
        errorMessageDiv.textContent = '';
        sentimentSpan.className = ''; // Remove any previous styling
        
        if (!text) {
            errorMessageDiv.textContent = 'Please enter some text to analyze.';
            errorMessageDiv.style.display = 'block';
            return; // Stop if text is empty
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text }) // Send text as JSON
            });

            const data = await response.json(); // Parse the JSON response

            if (response.ok) { // Check if the HTTP status code is in the 200s
                sentimentSpan.textContent = data.sentiment;
                confidenceSpan.textContent = `${data.confidence}%`;

                // Add styling based on sentiment
                if (data.sentiment === 'positive') {
                    sentimentSpan.className = 'positive';
                } else if (data.sentiment === 'negative') {
                    sentimentSpan.className = 'negative';
                }
            } else {
                // Handle API errors (e.g., 400 Bad Request from Flask)
                errorMessageDiv.textContent = `Error: ${data.message || 'Unknown error'}`;
                errorMessageDiv.style.display = 'block';
            }

        } catch (error) {
            // Handle network errors or other unexpected issues
            errorMessageDiv.textContent = `An error occurred while connecting to the server: ${error.message}`;
            errorMessageDiv.style.display = 'block';
            console.error('Fetch error:', error);
        }
    });
});