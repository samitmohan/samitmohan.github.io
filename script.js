document.addEventListener('DOMContentLoaded', () => {
    const mainContainer = document.getElementById('main-container');
    const yesMessageContainer = document.getElementById('yes-message-container');
    
    const yesBtn1 = document.getElementById('yes-btn-1');
    const yesBtn2 = document.getElementById('yes-btn-2');
    const noBtn = document.getElementById('no-btn');
    
    const noReasonContainer = document.getElementById('no-reason-container');
    const reasonInput = document.getElementById('reason-input');
    const submitReasonBtn = document.getElementById('submit-reason-btn');

    const showYesMessage = () => {
        mainContainer.classList.add('hidden');
        yesMessageContainer.classList.remove('hidden');
    };

    const showNoForm = () => {
        noReasonContainer.classList.remove('hidden');
    };

    const handleSubmitReason = () => {
        const reason = reasonInput.value.trim();
        if (reason.length < 20) {
            alert('That\'s not a good enough reason. Please write an elaborate believable lie.');
        } else {
            alert('Your justification will be sent for review');
            window.location.href = `mailto:samitmohan@gmail.com?subject=A reason for saying no...&body=${encodeURIComponent(reason)}`;
        }
    };

    yesBtn1.addEventListener('click', showYesMessage);
    yesBtn2.addEventListener('click', showYesMessage);
    noBtn.addEventListener('click', showNoForm);
    submitReasonBtn.addEventListener('click', handleSubmitReason);
});
