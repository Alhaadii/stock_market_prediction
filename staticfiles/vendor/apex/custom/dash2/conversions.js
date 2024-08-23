var options = {
	chart: {
		height: 260,
		type: "bar",
		toolbar: {
			show: false,
		},
	},
	plotOptions: {
		bar: {
			columnWidth: "50%",
			borderRadius: 8,
			dataLabels: {
				position: "center",
			},
		},
	},
	series: [
		{
			name: "Designation",
			data: [50, 60, 70, 80, 90],
		},
	],
	legend: {
		show: false,
	},
	xaxis: {
		categories: ["A1", "A2", "A3", "A4", "A5"],
		axisBorder: {
			show: false,
		},
		yaxis: {
			show: false,
		},

		tooltip: {
			enabled: true,
		},
		labels: {
			show: true,
			rotate: -45,
			rotateAlways: true,
		},
	},
	grid: {
		borderColor: "#b7c6d8",
		strokeDashArray: 5,
		xaxis: {
			lines: {
				show: true,
			},
		},
		yaxis: {
			lines: {
				show: false,
			},
		},
		padding: {
			top: 0,
			right: 10,
			left: 20,
			bottom: -20,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return val;
			},
		},
	},
	colors: ["#2f49d0"],
};
var chart = new ApexCharts(document.querySelector("#conversions"), options);
chart.render();
