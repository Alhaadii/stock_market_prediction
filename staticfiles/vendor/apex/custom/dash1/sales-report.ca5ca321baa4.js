var options = {
	chart: {
		height: 350,
		type: "bar",
		stacked: true,
		toolbar: {
			show: false,
		},
	},
	plotOptions: {
		bar: {
			columnWidth: "20%",
			borderRadius: 10,
		},
	},
	dataLabels: {
		enabled: false,
	},
	series: [
		{
			name: "Balance",
			data: [52, 73, 34, 66, 82, 49, 38, 59, 44, 86, 30, 60],
		},
		{
			name: "3",
			data: [32, 43, 24, 36, 33, 29, 18, 29, 14, 56, 20, 30],
		},
	],
	legend: {
		show: false,
	},
	xaxis: {
		categories: [
			"Jan",
			"Feb",
			"Mar",
			"Apr",
			"May",
			"Jun",
			"Jul",
			"Aug",
			"Sep",
			"Oct",
			"Nov",
			"Dec",
		],
		axisBorder: {
			show: false,
		},
		yaxis: {
			show: false,
		},
		tooltip: {
			enabled: true,
		},
		fill: {
			opacity: 1,
		},
		labels: {
			show: true,
			rotate: -45,
			rotateAlways: true,
		},
	},
	grid: {
		borderColor: "#dae1ea",
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
			right: 0,
			bottom: 0,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return val + " million";
			},
		},
	},
	colors: [
		"#2f49d0",
		"#f72b2b",
		"#7d7ad2",
		"#5854c5",
		"#3f3aab",
		"#312d85",
		"#ea95bf",
		"#b5d085",
	],
};
var chart = new ApexCharts(document.querySelector("#salesReport"), options);
chart.render();
