queue()
    .defer(d3.tsv, "predictions.csv")
    .await(showPredictions);

// set initial value for global variables
var selectedCtry = "Afghanistan"
var selectedCmd = "Abrasive cloths, papers etc (including sandpaper)"
var mapMode = 'select country';
var target = 1 //{1: exports, 2: comparative advantage}


function showPredictions(error, data) {
    if(error) { console.log(error); }

  data.forEach(function(d) {
        d.probability = +d.proba_pred;
        d.target = +d.target;
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

 function Data(data, target) {
    return data.filter(function(d){
          return d.target == target;})
  };

  function countryData(ctry, target) {
    return Data(data, target).filter(function(d){
          return d.Country == ctry;})
  };

  var countryDataCode = function (ctry) {
    return Data(data, target).filter(function(d){
          return d.origin == ctry;})
  };

  var countryNameFromCode = function (code) {
         var slice = countryDataCode(code);
         if (slice[0]!=undefined) {var name = slice[0].Country}
         else {var name = "n/a"};
         return name
  };

    var countryCodeFromName = function (name) {
         var slice = countryData(name, target);
         return slice[0].origin;
  };


  function cmdData(cmd, target) {
    return Data(data, target).filter(function(d){
          return d.product == cmd;})
  };

  function ctryTitle(Country, target) {
     if (target == 1) {
        return "Products ".concat(Country).concat(" may start exporting");};
     if (target == 2) {
        return "Products ".concat(Country).concat(" may develop comparative advantage in");};
  }

  function productTitle(cmd, target) {
        if (target == 1) {
            return "Countries likely to start exporting ".concat(cmd);};
        if (target == 2) {
            return "Countries likely to develop comparative advantage in ".concat(cmd);};
  }



  function treeNote(mode, selection, target){
      if (mode == 'select country'){
        var count = countryData(selection, target).length.toString()
        if (target == 2) {
            var string = "There are ".concat(count).concat(" products ").concat(selection).concat(" may develop comparative advantage in.");
        } else {
        if (count == 1) {var string = "There is only 1 product ".concat(selection).concat(" has not exported. Try predicting comparative advantage.");
        }
        if (count >1 && count < 20) {var string = "There are only ".concat(count).concat(" products ").concat(selection).concat(" has not exported. Try predicting comparative advantage.");
        }
        if (count >= 20) {var string = "There are ".concat(count).concat(" products ").concat(selection).concat(" has not exported.");};
      }}
      if (mode == 'select product'){
        var count = cmdData(selection, target).length.toString()
        if (target==2) {
            if (count==1) {var string = count.concat(" country may develop comparative advantage in ").concat(selection);}
            else {var string = count.concat(" countries may develop comparative advantage in ").concat(selection);}
        } else {
        if (count == 1) {var string = "Only 1 country has not exported ".concat(selection).concat('.');
        } else {var string = count.concat(" countries have not exported ").concat(selection).concat('.');};
      }}
      document.getElementById('tree-notes').innerHTML = string;
    };
  treeNote('select country', selectedCtry, target);

  var tree = d3plus.viz()
    .container("#tree")
    .title(ctryTitle(selectedCtry, target))
    .data(countryData(selectedCtry, target))
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

function updateTree(mode, selection, target){
    if (mode == 'select country'){
        var ctryD = countryData(selection, target);
        tree.id(['category','SubCategory','product'])
        tree.title(ctryTitle(selection, target));
        tree.data(ctryD).draw();
    }
    if (mode == 'select product'){
        var cmdD = cmdData(selection, target);
        tree.id('Country')
        tree.title(productTitle(selection, target));
        tree.data(cmdD).draw();
    }
    treeNote(mode, selection, target);
}

  var table = d3plus.viz()
    .container("#table")
    .type("table")
    .id('cmd')
    .cols({"value": ['category','product','probability'],"index":false})
	.shape("square")
	.messages("Loading...")
	.font({"align":"left"})

function updateTable(mode, selection, target){
    if (mode == 'select country'){
        var ctryD = countryData(selection, target);
        var height = 30*(ctryD.length+1); // 30 pixels for each row including header
        table.id('cmd')
        table.cols({"value": ['category','product','probability'],"index":false})
        table.title(ctryTitle(selection, target));
        table.height(height);
        table.data(ctryD)//.draw();
    }
    if (mode == 'select product'){
        var cmdD = cmdData(selection, target);
        var height = 30*(cmdD.length+1); // 30 pixels for each row including header
        table.id('origin')
        table.cols({"value": ['Country','probability'],"index":false})
        table.title(productTitle(selection, target));
        table.height(height);
        table.data(cmdD)//.draw();
    }
}
updateTable('select country', selectedCtry);

  var map = d3plus.viz()
    .container("#map")
    .title("Select a country")
    .data(Data(data, target))
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
    .ui([{
        "method": function(value) {
            target = value;
            console.log(target)
            console.log(mapMode)
            console.log(selectedCmd)
            console.log(selectedCtry)
            if (mapMode=='select country') {
                updateTree(mapMode, selectedCtry, target);
                updateTable(mapMode, selectedCtry, target);
            }
            if (mapMode=='select product') {
                updateTree(mapMode, selectedCmd, target);
                updateTable(mapMode, selectedCmd, target);
            }
            updateMap(mapMode, target, selectedCtry);},
        "type": 'toggle',
        "value": [{'Predict exports': 1}, {'Predict comparative advantage': 2}]
        },
        {
        "method": function(value) {
            if (value=='Browse by country'){
                mapMode ='select country'
                updateTree(mapMode, selectedCtry, target);
                updateTable(mapMode, selectedCtry, target);
                updateMap(mapMode, target, selectedCtry)
                };
            if (value=='Browse by product'){
                mapMode ='select product';
                updateTree(mapMode, selectedCmd, target);
                updateTable(mapMode, selectedCmd, target);
                updateMap(mapMode, target, selectedCmd)
                }
            updateMap(mapMode, target, selectedCtry);
            console.log(target);
            console.log(mapMode);
            console.log(selectedCmd);
            console.log(selectedCtry);
            },
        "value": ['Browse by country', 'Browse by product']
        },
        {
        "width": 250,
        "method": function(value, map) {
            selectedCmd = value
            mapMode = 'select product'
            //map.title(productTitle(selectedCmd, target));
            //map.data(cmdData(selectedCmd, target));
            updateMap('select product', target, selectedCmd);
            updateTree(mapMode, selectedCmd, target);
            updateTable(mapMode, selectedCmd, target);
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
                selectedCtry = ctry_name
                mapMode = 'select country';
                updateTree(mapMode, selectedCtry, target);
                updateTable(mapMode, selectedCtry, target);
                map.title("Click selected country to zoom out");
                }
            console.log(target);
            console.log(mapMode);
            console.log(selectedCmd);
            console.log(selectedCtry);
            };


    function updateMap(mode, target, selection){
         if (mode =='select country') {
            map.data(Data(data, target));
            map.title('Select a country');
            map.color("Country");
            map.draw();
            updateTree(mode, selection, target);
            updateTable(mode, selection, target);
        };
        if (mode =='select product') {
            map.title(productTitle(selection, target));
            map.data(cmdData(selection, target));
            map.color("probability");
            map.color("probability");
            map.tooltip("probability");
            map.draw();

        };
    }

};

