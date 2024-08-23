document.addEventListener("DOMContentLoaded", function () {
	var calendarEl = document.getElementById("dayGrid");
	var calendar = new FullCalendar.Calendar(calendarEl, {
		headerToolbar: {
			left: "prevYear,prev,next,nextYear today",
			center: "title",
			right: "dayGridMonth,dayGridWeek,dayGridDay",
		},
		initialDate: "2022-10-12",
		navLinks: true, // can click day/week names to navigate views
		editable: true,
		dayMaxEvents: true, // allow "more" link when too many events
		events: [
			{
				title: "All Day Event",
				start: "2022-10-01",
				color: "#2f49d0",
			},
			{
				title: "Long Event",
				start: "2022-10-07",
				end: "2022-10-10",
				color: "#445bd5",
			},
			{
				groupId: 999,
				title: "Birthday",
				start: "2022-10-09T16:00:00",
				color: "#596dd9",
			},
			{
				groupId: 999,
				title: "Birthday",
				start: "2022-10-16T16:00:00",
				color: "#2a42bb",
			},
			{
				title: "Conference",
				start: "2022-10-11",
				end: "2022-10-13",
				color: "#2f49d0",
			},
			{
				title: "Meeting",
				start: "2022-10-14T10:30:00",
				end: "2022-10-14T12:30:00",
				color: "#6d80de",
			},
			{
				title: "Lunch",
				start: "2022-10-16T12:00:00",
				color: "#8292e3",
			},
			{
				title: "Meeting",
				start: "2022-10-18T14:30:00",
				color: "#2f49d0",
			},
			{
				title: "Interview",
				start: "2022-10-21T17:30:00",
				color: "#2a42bb",
			},
			{
				title: "Meeting",
				start: "2022-10-22T20:00:00",
				color: "#596dd9",
			},
			{
				title: "Birthday",
				start: "2022-10-13T07:00:00",
				color: "#263aa6",
			},
			{
				title: "Click for Google",
				url: "http://google.com/",
				start: "2022-10-28",
				color: "#213392",
			},
			{
				title: "Interview",
				start: "2022-10-20",
				color: "#8292e3",
			},
			{
				title: "Product Launch",
				start: "2022-10-29",
				color: "#2a42bb",
			},
			{
				title: "Leave",
				start: "2022-10-25",
				color: "#6d80de",
			},
		],
	});

	calendar.render();
});
