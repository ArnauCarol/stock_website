var ctx = document.getElementById('stockChart').getContext('2d');

var stockChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: stockDates,
        datasets: [{
            label: 'Stock Price',
            data: stockPrices,
            borderColor: 'blue',
            backgroundColor: 'rgba(0, 0, 255, 0.1)',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'month'
                }
            },
            y: {
                beginAtZero: false
            }
        }
    }
});
