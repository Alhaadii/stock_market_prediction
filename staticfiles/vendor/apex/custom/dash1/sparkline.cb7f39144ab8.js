// Chart 1
var options1 = {
	chart: {
		height: 110,
		width: "100%",
		type: "area",
		toolbar: {
			show: false,
		},
	},
	dataLabels: {
		enabled: false,
	},
	stroke: {
		curve: "smooth",
		width: 4,
	},
	series: [
		{
			name: "Off Shore",
			data: [10, 30, 15, 40, 15, 40, 70],
		},
	],
	grid: {
		borderColor: "#dae1ea",
		strokeDashArray: 5,
		xaxis: {
			lines: {
				show: false,
			},
		},
		yaxis: {
			lines: {
				show: false,
			},
		},
		padding: {
			top: -20,
			right: 0,
			bottom: 0,
			left: -10,
		},
	},
	xaxis: {
		labels: {
			show: false,
		},
	},
	yaxis: {
		labels: {
			show: false,
		},
	},
	colors: ["#2f49d0"],
	markers: {
		size: 0,
		opacity: 0.3,
		colors: ["#2f49d0"],
		strokeColor: "#ffffff",
		strokeWidth: 2,
		hover: {
			size: 7,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return +val + "k";
			},
		},
	},
};
var chart = new ApexCharts(document.querySelector("#revenue"), options1);
chart.render();

// Chart 2
var options2 = {
	chart: {
		height: 110,
		width: "100%",
		type: "area",
		toolbar: {
			show: false,
		},
	},
	dataLabels: {
		enabled: false,
	},
	stroke: {
		curve: "smooth",
		width: 4,
	},
	series: [
		{
			name: "Off Shore",
			data: [10, 30, 15, 40, 15, 40, 70],
		},
	],
	grid: {
		borderColor: "#dae1ea",
		strokeDashArray: 5,
		xaxis: {
			lines: {
				show: false,
			},
		},
		yaxis: {
			lines: {
				show: false,
			},
		},
		padding: {
			top: -20,
			right: 0,
			bottom: 0,
			left: -10,
		},
	},
	xaxis: {
		labels: {
			show: false,
		},
	},
	yaxis: {
		labels: {
			show: false,
		},
	},
	colors: ["#2f49d0"],
	markers: {
		size: 0,
		opacity: 0.3,
		colors: ["#2f49d0"],
		strokeColor: "#ffffff",
		strokeWidth: 2,
		hover: {
			size: 7,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return +val + "k";
			},
		},
	},
};
var chart = new ApexCharts(document.querySelector("#revenue2"), options2);

chart.render();

// Chart 3
var options3 = {
	chart: {
		height: 110,
		width: "100%",
		type: "area",
		toolbar: {
			show: false,
		},
	},
	dataLabels: {
		enabled: false,
	},
	stroke: {
		curve: "smooth",
		width: 4,
	},
	series: [
		{
			name: "Off Shore",
			data: [10, 30, 15, 40, 15, 40, 70],
		},
	],
	grid: {
		borderColor: "#dae1ea",
		strokeDashArray: 5,
		xaxis: {
			lines: {
				show: false,
			},
		},
		yaxis: {
			lines: {
				show: false,
			},
		},
		padding: {
			top: -20,
			right: 0,
			bottom: 0,
			left: -10,
		},
	},
	xaxis: {
		labels: {
			show: false,
		},
	},
	yaxis: {
		labels: {
			show: false,
		},
	},
	colors: ["#2f49d0"],
	markers: {
		size: 0,
		opacity: 0.3,
		colors: ["#2f49d0"],
		strokeColor: "#ffffff",
		strokeWidth: 2,
		hover: {
			size: 7,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return +val + "k";
			},
		},
	},
};
var chart = new ApexCharts(document.querySelector("#revenue3"), options3);

chart.render();

// Chart 4
var options4 = {
	chart: {
		height: 110,
		width: "100%",
		type: "area",
		toolbar: {
			show: false,
		},
	},
	dataLabels: {
		enabled: false,
	},
	stroke: {
		curve: "smooth",
		width: 4,
	},
	series: [
		{
			name: "Off Shore",
			data: [10, 30, 15, 40, 15, 40, 70],
		},
	],
	grid: {
		borderColor: "#dae1ea",
		strokeDashArray: 5,
		xaxis: {
			lines: {
				show: false,
			},
		},
		yaxis: {
			lines: {
				show: false,
			},
		},
		padding: {
			top: -20,
			right: 0,
			bottom: 0,
			left: 0,
		},
	},
	xaxis: {
		labels: {
			show: false,
		},
	},
	yaxis: {
		labels: {
			show: false,
		},
	},
	colors: ["#f72b2b"],
	markers: {
		size: 0,
		opacity: 0.3,
		colors: ["#f72b2b"],
		strokeColor: "#ffffff",
		strokeWidth: 2,
		hover: {
			size: 7,
		},
	},
	tooltip: {
		y: {
			formatter: function (val) {
				return +val + "k";
			},
		},
	},
};
var chart = new ApexCharts(document.querySelector("#revenue4"), options4);

chart.render();
