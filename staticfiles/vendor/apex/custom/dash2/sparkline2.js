// Sparkline 1
var options1 = {
	series: [80],
	chart: {
		type: "radialBar",
		width: 90,
		height: 90,
		sparkline: {
			enabled: true,
		},
	},
	dataLabels: {
		enabled: false,
	},
	colors: ["#2f49d0"],
	plotOptions: {
		radialBar: {
			hollow: {
				margin: 0,
				size: "20%",
			},
			track: {
				background: "#cdd1dd",
				margin: 0,
			},
			dataLabels: {
				show: false,
			},
		},
	},
};
var chart1 = new ApexCharts(document.querySelector("#radial1"), options1);
chart1.render();

// Sparkline 2
var options2 = {
	series: [70],
	chart: {
		type: "radialBar",
		width: 90,
		height: 90,
		sparkline: {
			enabled: true,
		},
	},
	dataLabels: {
		enabled: false,
	},
	colors: ["#2f49d0"],
	plotOptions: {
		radialBar: {
			hollow: {
				margin: 0,
				size: "20%",
			},
			track: {
				background: "#cdd1dd",
				margin: 0,
			},
			dataLabels: {
				show: false,
			},
		},
	},
};
var chart2 = new ApexCharts(document.querySelector("#radial2"), options2);
chart2.render();

// Sparkline 3
var options3 = {
	series: [60],
	chart: {
		type: "radialBar",
		width: 90,
		height: 90,
		sparkline: {
			enabled: true,
		},
	},
	dataLabels: {
		enabled: false,
	},
	colors: ["#f72b2b"],
	plotOptions: {
		radialBar: {
			hollow: {
				margin: 0,
				size: "20%",
			},
			track: {
				background: "#cdd1dd",
				margin: 0,
			},
			dataLabels: {
				show: false,
			},
		},
	},
};
var chart3 = new ApexCharts(document.querySelector("#radial3"), options3);
chart3.render();
