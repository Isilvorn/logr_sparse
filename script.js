// defining the size and padding for the scatterplot
var w = 900,
    h = 600,
    pad = 20,
    left_pad = 100;

// selecting the correct portion of the page to print the scatterplot 
var svg = d3.select("#scatterplot")
        .append("svg")
        .attr("width", w)
        .attr("height", h);
 
var x = d3.scale.linear().domain([0.0, 1.0]).range([left_pad, w-pad]),
    y = d3.scale.linear().domain([0.0, 1.0]).range([h-pad*2, pad]);
 
var xAxis = d3.svg.axis().scale(x).orient("bottom")
        .ticks(10)
        .tickFormat(function (fpr, i) {
            return (fpr.toFixed(2));
        }),
    yAxis = d3.svg.axis().scale(y).orient("left")
        .ticks(10)
        .tickFormat(function (tpr, i) {
            return (tpr.toFixed(2));
        });
 
svg.append("g")
    .attr("class", "axis")
    .attr("transform", "translate(0, "+(h-pad)+")")
    .call(xAxis);
 
svg.append("g")
    .attr("class", "axis")
    .attr("transform", "translate("+(left_pad-pad)+")", 0)
    .call(yAxis);
 
svg.append("text")
    .attr("class", "loading")
    .text("Loading ...")
    .attr("x", function () { return w/2; })
    .attr("y", function () { return h/2-5; });
 
//var max_r = d3.max(punchcard_data.map(
//                   function (d) { return d[2]; })),
//        r = d3.scale.linear()
//            .domain([0, d3.max(punchcard_data, function (d) { return d[2]; })])
//            .range([0, 12]);
 
var r = d3.scale.linear()
          .domain([1.0, 0])
          .range([5, 8]);

var scatter_data = setdata();
svg.selectAll(".loading").remove();
 
svg.selectAll("circle")
    .data(scatter_data)
    .enter()
    .append("circle")
    .attr("class", "circle")
    .attr("cx", function (d) { return x(d[0]); })
    .attr("cy", function (d) { return y(d[1]); })
    .transition()
    .duration(800)
    .attr("r", function (d) { return r(d[2]); });
