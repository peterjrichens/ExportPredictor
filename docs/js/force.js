
var menu = d3.select("#menu1");

var path_hash = {
        '1': "json/rca.json",
        '2': "json/imports.json",
        '3': "json/export_destination.json",
        '4': "json/import_origin.json",
        '5': "json/intensity.json",
        '6': "json/distance.json"
};

function change_source() {

    var menu_value = menu.node().options[menu.node().selectedIndex].value;
    var source = path_hash[menu_value];

    var w = 1100,
    h = 620,
    cr = 5, // circle radius
    fill = d3.scale.category20();

var vis = d3.select("#chart")
  .append("svg:svg")
    .attr("width", w)
    .attr("height", h);

d3.json(source, function(json) {
  var force = d3.layout.force()
      .charge(-80)
      .linkDistance(50)
      .gravity(0.1)
      .nodes(json.nodes)
      .links(json.links)
      .size([w, h])
      .start();

  var link = vis.selectAll(".link")
      .data(json.links)
    .enter().append("svg:line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return d.weight; })
      .attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  var node_drag = d3.behavior.drag()
        .on("dragstart", dragstart)
        .on("drag", dragmove)
        .on("dragend", dragend);

  function dragstart(d, i) {
        force.stop();
  }

  function dragmove(d, i) {
        d.px += d3.event.dx;
        d.py += d3.event.dy;
        d.x += d3.event.dx;
        d.y += d3.event.dy;
        tick();
  }

  function dragend(d, i) {
        d.fixed = true;
        tick();
        force.resume();
        //window.setInterval(function() {d.fixed = false;}, 1000);
  }

  var color = d3.scale.category20();

  var node = vis.selectAll(".node")
      .data(json.nodes)
     .enter().append("g")
      .attr("class", "node")
      .attr("target", false)
      .call(node_drag);

  node.append("svg:circle")
      .style("fill", function(d) { return color(d.region); })
      .attr("data-legend",function(d) { return d.region})
      .attr("r", cr)
      .attr("id", function(d){
         return "c" + d.name.toLowerCase().replace(/\s+/g, '');})
      .on("mousedown", fade(.2))
      .on("mouseup", fade(1))
      .on("dblclick", function (d) {return d.fixed = false;})


  node.append("text")
      .attr("dx", 12)
      .attr("dy", ".35em")
      .text(function(d) { return d.name });


  node.append("svg:title")
      .text(function(d) {
      return d.name.concat("'s nearest neighbours are ",d.neighbours); });


  vis.style("opacity", 1e-6)
    .transition()
      .duration(1000)
      .style("opacity", 1);


    var linkedByIndex = {};
    json.links.forEach(function(d) {
        linkedByIndex[d.source.index + "," + d.target.index] = 1;
    });

    function isConnected(a, b) {
        return linkedByIndex[a.index + "," + b.index] || linkedByIndex[b.index + "," + a.index] || a.index == b.index;
    };

    function fade(opacity) {
        return function(d) {
            node.style("stroke-opacity", function(o) {
                thisOpacity = isConnected(d, o) ? 1 : opacity;
                this.setAttribute('fill-opacity', thisOpacity);
                return thisOpacity;
            });

            link.style("stroke-opacity", opacity).style("stroke-opacity", function(o) {
                return o.source === d || o.target === d ? 1 : opacity;
            });
        };
    };


    force.on("tick", tick);

    function tick() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("transform", function(d) {
        return "translate(" + Math.max(5, Math.min(w - 5, d.x)) + "," + Math.max(5, Math.min(h - 5, d.y)) + ")"; //bounded by box
    });

  };
    function agitate(d) {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("transform", function(d) {
        return "translate(" + Math.max(5, Math.min(w - 5, d.x)) + "," + Math.max(5, Math.min(h - 5, d.y)) + ")"; //bounded by box
    });

  };

    var legend = vis.selectAll(".legend")
        .data(color.domain())
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

    legend.append("rect")
        .attr("x", w - 18)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", color);

    legend.append("text")
        .attr("x", w - 24)
        .attr("y", 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(function(d) { return d; });
});

};
menu.on("change", function(d){
    d3.select("svg").remove(d);
    change_source(d);
});

change_source();

function resize_target(){
        if (the_circle) {the_circle.attr('r', cr)}; // reset any circles selected previously
        var userInput = document.getElementById("target_node");
        var the_circle = d3.select("#c"+userInput.value.toLowerCase().replace(/\s+/g, ''));
        the_circle.attr('r', 10);
 }

