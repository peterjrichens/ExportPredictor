queue()
    .defer(d3.tsv, "/data")
    .await(showPredictions);


function showPredictions(error, data) {
    if(error) { console.log(error); }

  data.forEach(function(d) {
        d.probability = +d.proba_pred;
        d.product = d.cmd_name;
        d.category = d.name_1dg;
        d.SubCategory = d.name_2dg;
        d.Country = d.ctry_name;
        });


  var ctries = d3.nest()
  .key(function(d) { return d.Country; })
  .entries(data);

 var ctry_list = ctries.map(i => i.key);

 var cmds = d3.nest()
  .key(function(d) { return d.product;})
  .entries(data);

 var cmd_list = cmds.map(i => i.key).sort()
 cmd_list.unshift('select a product'); //push placeholder to top


  var countryData = function (ctry) {
    return data.filter(function(d){
          return d.Country == ctry;})
  };

  var countryDataCode = function (ctry) {
    return data.filter(function(d){
          return d.origin == ctry;})
  };

  var countryNameFromCode = function (code) {
         var slice = countryDataCode(code);
         if (slice[0]!=undefined) {var name = slice[0].Country}
         else {var name = "n/a"};
         return name
  };

    var countryCodeFromName = function (name) {
         var slice = countryData(name);
         return slice[0].origin;
  };


  var cmdData = function (cmd) {
    return data.filter(function(d){
          return d.product == cmd;})
  };

  function ctryTitle(Country) {
        return "Products ".concat(Country).concat(" is most likely to start exporting");
  }

  function productTitle(cmd) {
        return "Countries most likely to start exporting ".concat(cmd);
  }

  var selectedCtry = ctry_list[Math.floor(Math.random()*ctry_list.length)]; // start with ramdom ctry in ctry_list

  function treeNote (mode, selection){
      if (mode == 'select country'){
        var count = (1229 - countryData(selection).length).toString()
        var string = selection.concat(" has already exported ").concat(count).concat(" out of 1229 products.");
      }
      if (mode == 'select product'){
        var count = (205 - cmdData(selection).length).toString()
        var string = count.concat(" countries have already exported ").concat(selection).concat('.');
      }
      document.getElementById('tree-notes').innerHTML = string;
    };
  treeNote('select country', selectedCtry);

  var selectedCmd = cmd_list[Math.floor(Math.random()*cmd_list.length)];


  var tree = d3plus.viz()
    .container("#tree")
    .title(ctryTitle(selectedCtry))
    .data(countryData(selectedCtry))
    .type("tree_map")
    .id(['category','SubCategory','product'])
    .size('probability')
    .tooltip({"share":false})
    .ui([{
        "value":[{"Show in table":1}],
        "type": "button",
        "method": function(value) {
            if (value.id==1){table.draw();}}
     }])
    .draw()

function updateTree(mode, selection){
    if (mode == 'select country'){
        var ctryD = countryData(selection);
        tree.id(['category','SubCategory','product'])
        tree.title(ctryTitle(selection));
        tree.data(ctryD).draw();
    }
    if (mode == 'select product'){
        var cmdD = cmdData(selection);
        tree.id('Country')
        tree.title(productTitle(selection));
        tree.data(cmdD).draw();
    }
    treeNote(mode, selection);
}

  var table = d3plus.viz()
    .container("#table")
    .type("table")
    .id('cmd')
    .cols({"value": ['category','product','probability'],"index":false})
	.shape("square")
	.messages("Loading...")
	.font({"align":"left"})

function updateTable(mode, selection){
    if (mode == 'select country'){
        var ctryD = countryData(selection);
        var height = 30*(ctryD.length+1); // 30 pixels for each row including header
        table.id('cmd')
        table.cols({"value": ['category','product','probability'],"index":false})
        table.title(ctryTitle(selection));
        table.height(height);
        table.data(ctryD)//.draw();
    }
    if (mode == 'select product'){
        var cmdD = cmdData(selection);
        var height = 30*(cmdD.length+1); // 30 pixels for each row including header
        table.id('origin')
        table.cols({"value": ['Country','probability'],"index":false})
        table.title(productTitle(selection));
        table.height(height);
        table.data(cmdD)//.draw();
    }
}
updateTable('select country', selectedCtry);

  var map = d3plus.viz()
    .container("#map")
    .title("Select a country")
    .data(data)
    .coords({
        "mute": ["010"], //hide Antartica
        "value":"https://d3js.org/world-50m.v1.json",
        "projection": "equirectangular"
        })
    .type("geo_map")
    .id("origin")
    .color("Country")
    .text({value: "Country", mute:["Kyrgyzstan","Mozambique"]}) //some issue with these two countries
    .focus(undefined, ctrySelectAction)
    .ui([
        {
        "method": function(value) {
            if (value=='Browse by country'){
                var mapMode ='select country'};

            if (value=='Browse by product'){
                var mapMode ='select product';
                map.title(productTitle(selectedCmd));
                map.data(cmdData(selectedCmd));
                updateTree(mapMode, selectedCmd);
                updateTable(mapMode, selectedCmd);
                }
            updateMap(mapMode);},
        "value": ['Browse by country', 'Browse by product']
        },
        {
        "width": 250,
        "method": function(value, map) {
            selectedCmd = value
            var mapMode = 'select product'
            map.title(productTitle(selectedCmd));
            map.data(cmdData(selectedCmd));
            updateMap('select product');
            updateTree(mapMode, selectedCmd);
            updateTable(mapMode, selectedCmd);
            },
        "type": "drop",
        "value": cmd_list
        }
        ])
        .draw()



    function ctrySelectAction(ctry_id){  // executed when user clicks on a country
            map.title("Select a country");
            if (map.focus != undefined) {
                map.focus(undefined).draw(); // zoom out
            };
            var ctry_name = countryNameFromCode(ctry_id);
            if (ctry_name != 'n/a'){
                var mapMode = 'select country';
                updateTree(mapMode, ctry_name);
                updateTable(mapMode, ctry_name);
                map.title("Click selected country to zoom out");
                }
            };


    function updateMap(mode){
         if (mode =='select country') {
            var selectedCtry = ctry_list[Math.floor(Math.random()*ctry_list.length)];
            map.title('Select a country');
            map.color("Country");
            map.data(data).draw();
            updateTree(mode, selectedCtry);
            updateTable(mode, selectedCtry);
        };
        if (mode =='select product') {
            map.color("probability");
            map.color("probability");
            map.tooltip("probability");
            map.draw();
        };
    }

};

