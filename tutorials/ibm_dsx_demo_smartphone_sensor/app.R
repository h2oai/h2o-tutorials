#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#


# ------------------------------------------------------------------------------
# Check and make sure H2O version 3.10.5.1 is installed
# ------------------------------------------------------------------------------

pkg_installed <- as.data.frame(installed.packages(), stringsAsFactors = FALSE)
row_h2o <- which(pkg_installed$Package == "h2o")
if (row_h2o != 0) ver_h2o <- pkg_installed[row_h2o,]$Version
if ((row_h2o == 0) | (ver_h2o != "3.10.5.1")) {

  # Install H2O version 3.10.5.1

  # The following two commands remove any previously installed H2O packages for R.
  if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
  if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

  # Next, we download packages that H2O depends on.
  pkgs <- c("statmod","RCurl","jsonlite")
  for (pkg in pkgs) {
    if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
  }

  # Now we download, install and initialize the H2O package for R.
  install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-vajda/1/R")

}


# ------------------------------------------------------------------------------
# Pre-load all R packages
# ------------------------------------------------------------------------------

suppressPackageStartupMessages(library(h2o))
suppressPackageStartupMessages(library(plotly))
suppressPackageStartupMessages(library(shiny))
suppressPackageStartupMessages(library(shinydashboard))
suppressPackageStartupMessages(library(DT))


# ------------------------------------------------------------------------------
# Pre-load all datasets and H2O models
# ------------------------------------------------------------------------------

# Start and connect to a H2O cluster (JVM)
h2o.init(nthreads = -1)

# Check if the datasets exist (locally)
chk_train <- suppressMessages(file.exists("./data/train.csv.gz"))
chk_test <- suppressMessages(file.exists("./data/test.csv.gz"))

# Import datasets (locally)
if (chk_train) hex_train <- h2o.importFile("./data/train.csv.gz")
if (chk_test) hex_test <- h2o.importFile("./data/test.csv.gz")

# Import datasets (from GitHub if they are not available locally)
if (!chk_train) hex_train <- h2o.importFile("https://github.com/woobe/h2o_demo_for_ibm_dsx/blob/master/data/train.csv.gz?raw=true")
if (!chk_test) hex_test <- h2o.importFile("https://github.com/woobe/h2o_demo_for_ibm_dsx/blob/master/data/test.csv.gz?raw=true")

# Load GBM model
model_gbm <- h2o.loadModel("./models/h2o_gbm")
model_pca <- h2o.loadModel("./models/h2o_pca")

# Extract all activities in test
d_test_activities <- as.data.frame(hex_test$activity)

# PCA
d_pca_train <- as.data.frame(h2o.predict(model_pca, hex_train))
d_pca_test <- as.data.frame(h2o.predict(model_pca, hex_test))
d_pca_train <- data.frame(activity = as.data.frame(hex_train$activity), d_pca_train)
d_pca_test <- data.frame(activity = as.data.frame(hex_test$activity), d_pca_test)



# ------------------------------------------------------------------------------
# Define UI for application
# ------------------------------------------------------------------------------

ui <- dashboardPage(

  skin = "blue",

  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## App Title
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dashboardHeader(title = "H2O and IBM DSX"),

  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## Sidebar content
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dashboardSidebar(
    sidebarMenu(
      menuItem("Predictions", tabName = "predict", icon = icon("play")),
      menuItem("Principle Component Analysis", tabName = "pca", icon = icon("area-chart"))
    )
  ), # End of dashboardSidebar


  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## Body content
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  dashboardBody(

    # Tabs
    tabItems(

      # ........................................................................
      # Tab predict
      # ........................................................................

      tabItem(
        tabName = "predict",
        h2("Comparison between Ground Truth and H2O Predictions"),
        br(),

        fluidRow(
          column(width = 3,
                 box(
                   title = "Control Panel", width = NULL, solidHeader = TRUE, status = "primary",

                   selectInput("activity", "Step 1 - Pick a Different Activity:",
                               c("WALKING" = "WALKING",
                                 "WALKING_DOWNSTAIRS" = "WALKING_DOWNSTAIRS",
                                 "WALKING_UPSTAIRS" = "WALKING_UPSTAIRS",
                                 "STANDING" = "STANDING",
                                 "SITTING" = "SITTING",
                                 "LAYING" = "LAYING"
                                 )),

                   p("Step 2 - Pick Another Random Sample:"),
                   submitButton("Go!", icon("refresh"))

                 )


          ),

          column(width = 9,

                 box(
                   title = "Random Sample from Test Dataset (as if it is new sensor data from a future smartphone user)",
                   width = NULL, status = "primary", solidHeader = FALSE,
                   DT::dataTableOutput("tbl_rand_samp")
                 ),

                 box(
                   title = "Predictions from H2O Model",
                   width = NULL, status = "primary", solidHeader = FALSE,
                   DT::dataTableOutput("tbl_pred")
                 )
          )
        )

      ), # End of predict tab

      # ........................................................................
      # PCA
      # ........................................................................

      tabItem(
        tabName = "pca",
        h2("Visualizing Principle Components from H2O Model"),
        br(),

        fluidRow(
          column(width = 3,
                 box(
                   title = "Control Panel", width = NULL, solidHeader = TRUE, status = "primary",

                   radioButtons("dataset", "Dataset:",
                                c("train" = "train",
                                  "test" = "test")),

                   radioButtons("x", "Variable for X-Axis:",
                                c("PC1" = "PC1",
                                  "PC2" = "PC2",
                                  "PC3" = "PC3",
                                  "PC4" = "PC4",
                                  "PC5" = "PC5"),
                                selected = "PC2"),

                   radioButtons("y", "Variable for Y-Axis:",
                                c("PC1" = "PC1",
                                  "PC2" = "PC2",
                                  "PC3" = "PC3",
                                  "PC4" = "PC4",
                                  "PC5" = "PC5"),
                                selected = "PC3"),

                   p("Refresh Graph:"),
                   submitButton("Go!", icon("refresh"))

                 )
          ),

          column(width = 9,

                 box(
                   title = "Interactive Plot",
                   width = NULL,
                   status = "primary", solidHeader = FALSE,
                   plotlyOutput("plot_pca")
                 )


          )
        )

      ) # End of pca tab

    ) # End of tabItems

  ) # End of dashboardBody

  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## End of UI
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

) # End of dashboardPage




# ------------------------------------------------------------------------------
# Define server logic
# ------------------------------------------------------------------------------

server <- function(input, output) {

  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## All Reactive Functions
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # Pick a random sample from test dataset based on activity chosen by user
  pick_rand_samp <- reactive({
    row_act <- which(d_test_activities == input$activity)
    row_samp <- sample(row_act, 1)
    df_samp <- as.data.frame(hex_test[row_samp, ])
    df_samp
  })

  # Using H2O GBM to make predictions
  make_pred <- reactive({
    hex_temp <- as.h2o(pick_rand_samp(), destination_frame = "hex_temp")
    yhat_temp <- h2o.predict(model_gbm, hex_temp)
    yhat_temp <- as.data.frame(yhat_temp)
    yhat_temp[, -1] <- round(yhat_temp[, -1], 6)
    colnames(yhat_temp) <- c("Predicted", "p(Laying)", "p(Sitting)", "p(Standing)",
                             "p(Walking)", "p(Walking Downstairs)", "p(Walking Upstairs")
    yhat_temp
  })


  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # Random Sample Table
  output$tbl_rand_samp <-
    DT::renderDataTable(t(pick_rand_samp()),
                        extensions = 'Scroller',
                        options = list(
                          dom = 't',
                          deferRender = TRUE,
                          scrollY = 200,
                          scroller = TRUE
                        ))

  # H2O Predictions Table
  output$tbl_pred <- DT::renderDataTable(make_pred(),
                                         options = list(dom = 't'))

  # Plotly PCA output
  output$plot_pca <- renderPlotly({

    if (input$dataset == "train") {

      temp_act <- d_pca_train$activity

      if (input$x == "PC1") temp_x <- d_pca_train$PC1
      if (input$x == "PC2") temp_x <- d_pca_train$PC2
      if (input$x == "PC3") temp_x <- d_pca_train$PC3
      if (input$x == "PC4") temp_x <- d_pca_train$PC4
      if (input$x == "PC5") temp_x <- d_pca_train$PC5

      if (input$y == "PC1") temp_y <- d_pca_train$PC1
      if (input$y == "PC2") temp_y <- d_pca_train$PC2
      if (input$y == "PC3") temp_y <- d_pca_train$PC3
      if (input$y == "PC4") temp_y <- d_pca_train$PC4
      if (input$y == "PC5") temp_y <- d_pca_train$PC5

    } else {

      temp_act <- d_pca_test$activity

      if (input$x == "PC1") temp_x <- d_pca_test$PC1
      if (input$x == "PC2") temp_x <- d_pca_test$PC2
      if (input$x == "PC3") temp_x <- d_pca_test$PC3
      if (input$x == "PC4") temp_x <- d_pca_test$PC4
      if (input$x == "PC5") temp_x <- d_pca_test$PC5

      if (input$y == "PC1") temp_y <- d_pca_test$PC1
      if (input$y == "PC2") temp_y <- d_pca_test$PC2
      if (input$y == "PC3") temp_y <- d_pca_test$PC3
      if (input$y == "PC4") temp_y <- d_pca_test$PC4
      if (input$y == "PC5") temp_y <- d_pca_test$PC5

    }

    # Create temp_df
    temp_df <- data.frame(activity = temp_act, x = temp_x, y = temp_y)

    # Create plotly plot
    p <- plot_ly(data = temp_df, x = ~x, y = ~y, color = ~activity,
                 type = "scatter", mode = "markers", marker = list(size = 3)) %>%
      layout(yaxis = list(title = input$y), xaxis = list(title = input$x)) %>%
      layout(title = "Visualizing Principle Components")
    p

  })


} # End of server


# ------------------------------------------------------------------------------
# Run the application
# ------------------------------------------------------------------------------

shinyApp(ui = ui, server = server)

