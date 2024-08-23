var options = {
	chart: {
		height: 350,
		type: "radialBar",
		toolbar: {
			show: false,
		},
	},
	plotOptions: {
		radialBar: {
			hollow: {
				size: "40%",
			},
			dataLabels: {
				name: {
					fontSize: "12px",
					fontColor: "black",
				},
				value: {
					fontSize: "21px",
				},
				total: {
					show: true,
					label: "Visitors",
					formatter: function (w) {
						// By default this function returns the average of all series. The below is just an example to show the use of custom formatter function
						return "250";
					},
				},
			},
			track: {
				background: "#f4f7fe",
			},
		},
	},
	series: [80, 70],
	labels: ["Visitors", "Clicks"],
	colors: ["#2f49d0", "#f72b2b"],
};

var chart = new ApexCharts(document.querySelector("#visitors"), options);
chart.render();
