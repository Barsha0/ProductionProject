document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const linkInput = document.querySelector('input[name="linkInput"]');
    const submitButton = document.querySelector('input[name="btnCheck"]');
    const messageText = document.getElementById("messageText")

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        const url = linkInput.value;

        if (url) {
            try {
                const response = await fetch('http://127.0.0.1:8000/checker/checker/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ url: url })
                });

                const result = await response.json();
                if(result.isSafeUrl){
                    messageText.innerHTML = "The provided url is safe"
                }else{
                    messageText.innerHTML = "The provided url is malicious"
                }
            } catch (error) {
                messageText.innerHTML = "An error occurred. Please try again"
                console.log(error)
            }
        } else {
            messageText.innerHTML = "Please enter an url"
        }
    });
});